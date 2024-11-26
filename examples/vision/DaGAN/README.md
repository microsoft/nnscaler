This example demonstrates a GAN-like vision model. The nnscaler trainer assumes there is only one end-to-end module that needs to be parallelized. However, GAN-like models always have both a generator and a discriminator. Here, you will learn how to run your code without the nnscaler trainer, and how to parallelize, synchronize, and update modules during training.

In this example, both `GeneratorFullModel` and `DiscriminatorFullModel` contain the same keypoint detector, generator, and discriminator modules. A module cannot be parallelized multiple times, so keypoint detector, generator, and discriminator must be parallelized separately. Separate synchronization and updates are also needed during training.


# clone CVPR2022-DaGAN repository from github
```
cd MagicCube/examples/vision/DaGAN
git clone https://github.com/harlanhong/CVPR2022-DaGAN.git
```

# Install dependent packages
```
mv CVPR2022-DaGAN CVPR2022_DaGAN
cd CVPR2022_DaGAN
pip install --ignore-installed -r requirements.txt
cd ..
export PYTHONPATH=$PYTHONPATH:CVPR2022_DaGAN
```

# run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master_port=12348 run.py --config vox-adv-256.yaml --name DaGAN --batchsize 8 --kp_num 15 --generator DepthAwareGenerator
```