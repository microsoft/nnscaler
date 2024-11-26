import torch
import CVPR2022_DaGAN.depth as depth
from CVPR2022_DaGAN.modules.model import ImagePyramide
from CVPR2022_DaGAN.modules.model import Vgg19
from CVPR2022_DaGAN.modules.model import GeneratorFullModel
from CVPR2022_DaGAN.modules.model import DiscriminatorFullModel
from CVPR2022_DaGAN.modules.model import Transform
from CVPR2022_DaGAN.modules.model import detach_kp
import torch.nn.functional as F

class GeneratorFullModel_NNSCALER(GeneratorFullModel):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    What has been changed:
        1. Replace train_params by config in __init__, which include all the content in yaml, original get scale 
        by self.discriminator.module.scales, but in nnscaler, there is no scales in discriminator.module
        2. Remove self.depth_encoder.load_state_dict() and self.depth_decoder.load_state_dict() in __init__, 
        which needs extra download, for this example, pretriained weights is not necessary.
        3. Remove passing in driving_depth in forward, generated = self.generator(...),
        because driving_depth is not used in generator forward, and pass in an unused argument is not allowed in nnscaler.
    """

    def __init__(self, kp_extractor, generator, discriminator, config, opt):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = config['train_params']
        self.scales = self.train_params['scales']
        self.disc_scales = config['model_params']['discriminator_params']['scales']
        self.pyramid = ImagePyramide(self.scales, config['model_params']['common_params']['num_channels'])
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.opt = opt
        self.loss_weights = self.train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        self.depth_encoder = depth.ResnetEncoder(50, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
        self.set_requires_grad(self.depth_encoder, False) 
        self.set_requires_grad(self.depth_decoder, False) 
        self.depth_decoder.eval()
        self.depth_encoder.eval()

    def forward(self, x):
        depth_source = None
        depth_driving = None
        outputs = self.depth_decoder(self.depth_encoder(x['source']))
        depth_source = outputs[("disp", 0)]
        outputs = self.depth_decoder(self.depth_encoder(x['driving']))
        depth_driving = outputs[("disp", 0)]
        
        if self.opt.use_depth:
            kp_source = self.kp_extractor(depth_source)
            kp_driving = self.kp_extractor(depth_driving)
        elif self.opt.rgbd:
            source = torch.cat((x['source'],depth_source),1)
            driving = torch.cat((x['driving'],depth_driving),1)
            kp_source = self.kp_extractor(source)
            kp_driving = self.kp_extractor(driving)
        else:
            kp_source = self.kp_extractor(x['source'])
            kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            if self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transformed_kp = self.kp_extractor(depth_transform)
            elif self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transform_img = torch.cat((transformed_frame,depth_transform),1)
                transformed_kp = self.kp_extractor(transform_img)
            else:
                transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value


        if self.loss_weights['kp_distance']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            source_dist_loss = (-torch.sign((torch.sqrt((sk*sk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            driving_dist_loss = (-torch.sign((torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            # driving_dist_loss = (torch.sign(1-(torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()))+1).mean()
            value_total = self.loss_weights['kp_distance']*(source_dist_loss+driving_dist_loss)
            loss_values['kp_distance'] = value_total
        if self.loss_weights['kp_prior']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            dis_loss = torch.relu(0.1-torch.sqrt((sk*sk).sum(-1)+1e-8))+torch.relu(0.1-torch.sqrt((dk*dk).sum(-1)+1e-8))
            bs,nk,_=kp_source['value'].shape
            scoor_depth = F.grid_sample(depth_source,kp_source['value'].view(bs,1,nk,-1))
            dcoor_depth = F.grid_sample(depth_driving,kp_driving['value'].view(bs,1,nk,-1))
            sd_loss = torch.abs(scoor_depth.mean(-1,keepdim=True) - kp_source['value'].view(bs,1,nk,-1)).mean()
            dd_loss = torch.abs(dcoor_depth.mean(-1,keepdim=True) - kp_driving['value'].view(bs,1,nk,-1)).mean()
            value_total = self.loss_weights['kp_distance']*(dis_loss+sd_loss+dd_loss)
            loss_values['kp_distance'] = value_total


        if self.loss_weights['kp_scale']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            if self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                pred = torch.cat((generated['prediction'],depth_pred),1)
                kp_pred = self.kp_extractor(pred)
            elif self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                kp_pred = self.kp_extractor(depth_pred)
            else:
                kp_pred = self.kp_extractor(generated['prediction'])

            pred_mean = kp_pred['value'].mean(1,keepdim=True)
            driving_mean = kp_driving['value'].mean(1,keepdim=True)
            pk = kp_source['value']-pred_mean
            dk = kp_driving['value']- driving_mean
            pred_dist_loss = torch.sqrt((pk*pk).sum(-1)+1e-8)
            driving_dist_loss = torch.sqrt((dk*dk).sum(-1)+1e-8)
            scale_vec = driving_dist_loss/pred_dist_loss
            bz,n = scale_vec.shape
            value = torch.abs(scale_vec[:,:n-1]-scale_vec[:,1:]).mean()
            value_total = self.loss_weights['kp_scale']*value
            loss_values['kp_scale'] = value_total
        if self.loss_weights['depth_constraint']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
            depth_pred = outputs[("disp", 0)]
            value_total = self.loss_weights['depth_constraint']*torch.abs(depth_driving-depth_pred).mean()
            loss_values['depth_constraint'] = value_total
        return loss_values, generated


class DiscriminatorFullModel_NNSCALER(DiscriminatorFullModel):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    What has been changed:
        1. Replace train_params by config in __init__, the same reason as GeneratorFullModel_NNSCALER
    """

    def __init__(self, kp_extractor, generator, discriminator, config):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = config['train_params']
        self.scales = config['model_params']['discriminator_params']['scales']

        self.pyramid = ImagePyramide(self.scales, config['model_params']['common_params']['num_channels'])
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = self.train_params['loss_weights']