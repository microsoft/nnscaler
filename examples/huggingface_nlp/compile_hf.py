#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from _collections_abc import MutableMapping
import os
import json
import sys
import subprocess
import traceback
import torch
import psutil
import logging
import time
import nnscaler
import inspect
import logging
import argparse
import warnings
import pathlib
from compile_interface import ModelCompiler, TraceCompileException, calcu_max_diff, logger

INFO_FNAME = 'info'     # info.log will log all the info and other loggers will be redirected to this file
TRIED_FNAME = 'tried'   # tried.log will log all the model names that have been tried
LOADED_FNAME = 'loaded' # loaded.log will log all the model names that have been successful loaded from huggingface
ERROR_FNAME = 'error'   # error.log will log all the error messages
EXPORT_FNAME = 'export' # models successful exported
EXPORT_ALIGNED_FNAME = 'export_aligned' # models exported and aligned with original model
TRACE_FNAME = 'trace'   # models successful traced
TRACE_ALIGNED_FNAME = 'trace_aligned'   # models traced and aligned with original model
COMPILE_FNAME = 'compile'   # models successful compiled
COMPILE_ALIGNED_FNAME = 'compile_aligned'   # models compiled and aligned with original model
TRAIN_FNAME = 'train'   # models successful trained
TRAIN_ALIGNED_FNAME = 'train_aligned'   # models trained and aligned with original model

FXMODULE_PARSER_WARNING_FNAME = 'FxModuleParser_Warning.log'    # log file for FxModuleParser warning
COMPILE_ERROR_JSON = 'error.json'   # error.json will store the error summary for all the models


warnings.filterwarnings("ignore")
torch.set_printoptions(edgeitems = 2)
text: str = "Huggingface is a really excellent project!"
########## define logger ##########
loggers = {}


def setup_logger(log_file, level = logging.INFO, need_timestamp = True):
    """Setup a logger for log_file
    """
    logger = logging.getLogger(str(log_file))
    logger.setLevel(level)
    # logger will only init once for one log_file
    if not logger.handlers:
        handler = logging.FileHandler(log_file, "a")
        if need_timestamp:
            formatter = logging.Formatter('%(asctime)s [PID %(process)d][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            handler.setLevel(level)
        logger.addHandler(handler)
    return logger


def logger_redirect(logger, to_logger_file, prefix = '', need_timestamp=True, level = logging.INFO) -> logging.FileHandler:
    """Add logger to another file
    """
    result_handler = logging.FileHandler(to_logger_file, 'a')
    if need_timestamp:
        formatter = logging.Formatter(f'%(asctime)s [PID %(process)d][%(levelname)s]: {prefix} %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(f'{prefix} %(message)s')
    result_handler.setFormatter(formatter)
    result_handler.setLevel(level)
    logger.addHandler(result_handler)
    return result_handler


def add_logger(log_dir, log_key, prefix = "", level = logging.INFO, need_timestamp = False):
    """Init a logger, redirect it to INFO_FNAME and add it to global loggers"""
    global loggers
    if log_key in loggers and loggers['log_key'] is not None:
        return
    _logger = setup_logger(log_dir / f'{log_key}.log', level, need_timestamp)
    logger_redirect(_logger, log_dir / f'{INFO_FNAME}.log', prefix = prefix)
    loggers[log_key] = _logger


def setup_loggers(log_dir, level = logging.INFO):
    """Setup loggers for compiling process"""
    info_logger = setup_logger(log_dir / f'{INFO_FNAME}.log', level, need_timestamp = True)
    loggers[INFO_FNAME] = info_logger
    add_logger(log_dir, TRIED_FNAME, prefix="model tried: ", level = level, need_timestamp = False)
    add_logger(log_dir, LOADED_FNAME, prefix="model loaded: ", level = level, need_timestamp = False)
    add_logger(log_dir, ERROR_FNAME, prefix="", level = level, need_timestamp = False)

######### define logger ##########

def print_memory_usage(prefix : str = ""):
    """Print current gpu memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    loggers[INFO_FNAME].debug("When " + prefix + f": Current memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_info = smi_output.strip().split('\n')
        gpu_mem_tuple = []
        for idx, mem in enumerate(memory_info):
            used, total = mem.split(', ')
            gpu_mem_tuple.append((idx, int(used) / 1024, int(total) / 1024))
        loggers[INFO_FNAME].debug(f"GPU memory usage (index, used-GB, total-GB): {gpu_mem_tuple}")
    except subprocess.CalledProcessError as e:
        loggers[INFO_FNAME].error("Can't execute nvidia-smi command:", e.output)
    except FileNotFoundError:
        loggers[INFO_FNAME].error("nvidia-smi command not found , make sure nvidia driver has been install successfully.")


def _prepare_hf_nlp_input(model, dummy_input):
    """Preprocess dummy_input for huggingface nlp models"""
    if isinstance(dummy_input, MutableMapping):
        dummy_input = dict(dummy_input)
    assert isinstance(dummy_input, dict)
    forward_signature = inspect.signature(model.forward)
    if 'decoder_input_ids' in forward_signature.parameters and 'decoder_input_ids' not in dummy_input:
        dummy_input['decoder_input_ids'] = dummy_input.get('input_ids', None)
    if 'token_type_ids' not in forward_signature.parameters and 'token_type_ids' in dummy_input:
        dummy_input.pop('token_type_ids', None)
    if 'attention_mask' in dummy_input:
        dummy_input.pop('attention_mask', None)
    return dummy_input


def dump_orged_errors(model_name, error_dict, log_path):
    """Dump error_dict to json file log_path, error_dict is a summary for error:model_name pairs"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    first_line = f"{exc_type.__name__}: {exc_value}"
    first_line = first_line.replace(model_name, r"{model_name}")

    if first_line in error_dict:
        error_dict[first_line]['model_name'].append(model_name)
        error_dict[first_line]['count'] += 1
    else:
        error_dict[first_line] = {"count": 1, 'model_name': [model_name]}   #, "example": exception_string

    error_dict = dict(sorted(error_dict.items(), key=lambda item: item[1]["count"], reverse=True))

    with open(log_path, 'w') as json_file:
        json.dump(error_dict, json_file, indent=4)


def load_error_summary(log_dir):
    """Load error_dict from COMPILE_ERROR_JSON in log_dir, this is for resume"""
    errors = {}
    if os.path.exists(log_dir / COMPILE_ERROR_JSON):
        with open(log_dir / COMPILE_ERROR_JSON, 'r') as json_file:
            errors = json.load(json_file)
        return errors
    else:
        return errors


def print_model_size(model, model_name):
    """Print model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    loggers[INFO_FNAME].info('model size: {:.3f}MB'.format(size_all_mb))
    loggers[INFO_FNAME].info(f"{model_name} has parameter: {sum(p.numel() for p in model.parameters())}")
    print_memory_usage(f"after load model {model_name}")


class HFModelLoader:
    """Load huggingface model, tokenizer, config by model_name"""
    def __init__(self, model_name, cache_dir, reduce=False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.reduce = reduce

    def _load_model_from_config(self, config):
        torch.manual_seed(0)
        def _get_auto_model_class(config):
            try:
                if config.architectures:
                    architecture = config.architectures[0]
                    if "CausalLM" in architecture:
                        return AutoModelForCausalLM
                return AutoModel
            except AttributeError:
                return AutoModel
        try:
            model = _get_auto_model_class(config).from_config(config, trust_remote_code=True)
            return model
        except Exception as e:
            raise e

    def _load_model_from_pretrain(self):
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True, resume_download = True)

    def load_hf_config(self):
        try:
            config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True, resume_download = True)
            return config
        except Exception:
            return None

    def load_hf_model(self, config):
        """Load huggingface model by config or pretrained
        """
        def _reduce_model_size(config):
            params_to_reduce =  ["n_layer", "n_layers", "num_hidden_layers", "num_decoder_layers", "num_heads"]
            for param in params_to_reduce:
                if hasattr(config, param) and getattr(config, param) is not None:
                    loggers[INFO_FNAME].info(f"set {param}: {getattr(config, param)} to 1")
                    setattr(config, param, 1)
            return config
        try:
            if not self.reduce:
                model = self._load_model_from_pretrain()
            else:
                config = _reduce_model_size(config)
                model = self._load_model_from_config(config)
            return model
        except Exception:
            if not self.reduce:
                loggers[INFO_FNAME].info("load model from pretrain failed, try by config")
                try:
                    model = self._load_model_from_config(config)
                    return model
                except Exception:
                    raise
            raise

    def load_hf_nlp_tokenizer(self):
        """load huggingface nlp tokenizer by model_name"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True, resume_download = True)
            return tokenizer
        except (OSError, ValueError):
            # The script uses just one of the seven tokenizers below, as we're only checking if the logits match for the same input.
            # BertTokenizerFast, CamembertTokenizerFast tokenizer, XLMRobertaTokenizerFast tokenizer, DistilBertTokenizerFast tokenizer
            # T5TokenizerFast tokenizer, RobertaTokenizerFast tokenizer, GPT2TokenizerFast tokenizer
            loggers[INFO_FNAME].info("loading pretrained tokenizer failed, use bert-base-uncased tokenizer instead")
            from transformers import BertTokenizerFast
            return BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir, trust_remote_code=True, resume_download = True)


class HFCompiler:
    def __init__(self, args):
        self.model_name = args.model_name
        self.cache_dir = args.cache_dir
        self.trace = args.trace
        self.compile = args.compile
        self.train = args.train
        self.reduce = args.reduce
        self.export = args.export
        self.log_dir = args.log_dir
        self.model_loader = HFModelLoader(self.model_name, self.cache_dir, self.reduce)

    def load_resources(self):
        self.config = self.model_loader.load_hf_config()
        loggers[INFO_FNAME].info(f"config: {self.config}")
        if self.config is not None:
            loggers[INFO_FNAME].info(f"{self.model_name} config loaded")
        self.tokenizer = self.model_loader.load_hf_nlp_tokenizer()
        loggers[INFO_FNAME].info(f"{self.model_name} Tokenizer loaded")
        self.model = self.model_loader.load_hf_model(self.config)
        print_model_size(self.model, self.model_name)
        loggers[LOADED_FNAME].info(f"{self.model_name}, {self.config.architectures if hasattr(self, 'config') and self.config else None}")

    def compile_hf_worker(self):
        try:
            start_time = time.time()
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                subprocess.run('rm -rf gencode*.py fullmodel.pt.* dist_param_map.pt', shell=True, check=True)
            # load config, tokenizer, model
            loggers[TRIED_FNAME].info(f"{self.model_name}")
            model_name = self.model_name

            self.load_resources()
            dummy_input = self.tokenizer(text, return_tensors="pt")

            # build dummy_input and forward
            dummy_input = _prepare_hf_nlp_input(self.model, dummy_input)
            self.model.eval()

            compiler = ModelCompiler(self.model, dummy_input, 'dp')
            if self.export and torch.distributed.get_rank() == 0:
                add_logger(self.log_dir, EXPORT_FNAME, prefix="model exported: ", level = logging.INFO, need_timestamp = False)
                add_logger(self.log_dir, EXPORT_ALIGNED_FNAME, prefix="model export aligned: ", level = logging.INFO, need_timestamp = False)
                emodel = compiler.export()
                max_diff = compiler.forward_diff(emodel)
                loggers[EXPORT_FNAME].info(f"{model_name}")
                loggers[INFO_FNAME].info(f"export max diff: {max_diff}")
                if max_diff <= 1e-5:
                    loggers[EXPORT_ALIGNED_FNAME].info(f"{model_name}")
                else:
                    loggers[ERROR_FNAME].error(f"{model_name} not aligned before and after export, max diff:{max_diff}")

            if self.trace and torch.distributed.get_rank() == 0:
                if self.export:
                    self.model = self.model_loader.load_hf_model(self.config)
                    print_model_size(self.model, self.model_name)
                add_logger(self.log_dir, TRACE_FNAME, prefix="model traced: ", level = logging.INFO, need_timestamp = False)
                add_logger(self.log_dir, TRACE_ALIGNED_FNAME, prefix="model trace aligned: ", level = logging.INFO, need_timestamp = False)
                t_model = compiler.trace()
                max_diff = compiler.forward_diff(t_model)
                if t_model:
                    loggers[TRACE_FNAME].info(f"{model_name}")
                loggers[INFO_FNAME].info(f"trace max diff: {max_diff}")
                if max_diff <= 1e-5:
                    loggers[TRACE_ALIGNED_FNAME].info(f"{model_name}")
                else:
                    loggers[ERROR_FNAME].error(f"{model_name} not aligned before and after trace, max diff:{max_diff}")
                del self.model
                torch.cuda.empty_cache()

            if self.compile or self.train:
                if self.export or self.trace:
                    # this model should be load again if traced before because the model will be changed during trace
                    self.model = self.model_loader.load_hf_model(self.config)
                    print_model_size(self.model, self.model_name)
                add_logger(self.log_dir, COMPILE_FNAME, prefix="model compiled: ", level = logging.INFO, need_timestamp = False)
                add_logger(self.log_dir, COMPILE_ALIGNED_FNAME, prefix="model compile aligned: ", level = logging.INFO, need_timestamp = False)
                p_model = compiler.parallel(self.model)
                max_diff = compiler.forward_diff(p_model)
                if p_model:
                    loggers[COMPILE_FNAME].info(f"{model_name}")
                loggers[INFO_FNAME].info(f"compile max diff: {max_diff}")
                if max_diff <= 1e-5:
                    loggers[COMPILE_ALIGNED_FNAME].info(f"{model_name}")
                else:
                    loggers[ERROR_FNAME].error(f"{model_name} not aligned before and after compile, max diff:{max_diff}")

                if self.train:
                    self.model = self.model_loader.load_hf_model(self.config)
                    add_logger(self.log_dir, TRAIN_FNAME, prefix="model trained: ", level = logging.INFO, need_timestamp = False)
                    add_logger(self.log_dir, TRAIN_ALIGNED_FNAME, prefix="model train aligned: ", level = logging.INFO, need_timestamp = False)
                    steps = 10
                    compile_loss = compiler.train(p_model, steps = steps)
                    compile_logit = p_model(**compiler.dummy_input)

                    origin_loss = compiler.train(self.model, steps = steps)
                    origin_logit = self.model(**compiler.dummy_input)

                    loggers[TRAIN_FNAME].info(f"{model_name}")

                    max_diff = calcu_max_diff(origin_logit, compile_logit)
                    if max_diff <= 1e-5:
                        loggers[TRAIN_ALIGNED_FNAME].info(f"{model_name}")
                    else:
                        loggers[ERROR_FNAME].error(f"{model_name} not aligned before and after train, max diff:{max_diff}")
        except TraceCompileException as e:
            # Exception will be cause from nnscaler.compile, or the program will be blocked
            if torch.distributed.get_rank() == 0:
                loggers[INFO_FNAME].error(f"fail when nnscaler.compile: {model_name}", exc_info=False)
                error_message = traceback.format_exc().strip() + "\n"
                loggers[ERROR_FNAME].error(f"{model_name}, {self.config.architectures if 'config' in locals() and self.config else None}, failed")
                loggers[ERROR_FNAME].error(error_message)
                dump_orged_errors(model_name, error_dict, self.log_dir / COMPILE_ERROR_JSON)
                import glob
                if not bool(glob.glob('gencode*.py')):
                    torch.distributed.barrier()
            raise
        except Exception as e:
            if torch.distributed.get_rank() == 0:
                loggers[INFO_FNAME].error(f"fail: {model_name}", exc_info=False)

                error_message = traceback.format_exc().strip() + "\n"
                loggers[ERROR_FNAME].error(f"{model_name}, {self.config.architectures if 'config' in locals() and self.config else None}, failed")
                loggers[ERROR_FNAME].error(error_message)
                dump_orged_errors(model_name, error_dict, self.log_dir / COMPILE_ERROR_JSON)
            raise
        finally:
            end_time = time.time()
            loggers[INFO_FNAME].info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='The script to compile huggingface nlp models')
    parser.add_argument('-d', '--cache_dir', default='/tmp/hf_cache', help='cache directory for config, tokenizer, model')
    parser.add_argument('-m', '--model_name', required=True, help='model name in huggingface')
    parser.add_argument('-t', '--trace', default=False, action=argparse.BooleanOptionalAction, help='do trace')
    parser.add_argument('-c', '--compile', default=True, action=argparse.BooleanOptionalAction, help='do compile')
    parser.add_argument('-tr', '--train', default=True, action=argparse.BooleanOptionalAction, help='train steps')
    parser.add_argument('-r', '--reduce', default=False, action=argparse.BooleanOptionalAction, help='whether reduce large models by setting layers to 1')
    parser.add_argument('-e', '--export', default=False, action=argparse.BooleanOptionalAction, help='torch.export models')
    parser.add_argument('-l', '--log_dir', default='~/hf_logs', help='log directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """ This script is for compile huggingface models and logs activities in argument of '--log_dir'.
        It will first load configuration, tokenizer, model from huggingface, then do export, trace, compile, train one by one and check alignment at each step.
        Among them, only "compile and train" is enabled by default, while "export" and "trace" are disabled by default.
        Now it only supports huggingface nlp models.
    usage:
        torchrun --nproc_per_node=1 --nnodes=1 compile_hf.py  -d <cache-dir> -m <model-name> -r -l <log-dir>
    """
    args = parse_arguments()
    if isinstance(args.log_dir, str):
        args.log_dir = pathlib.Path(os.path.expanduser(args.log_dir))
        args.cache_dir = pathlib.Path(os.path.expanduser(args.cache_dir))
    nnscaler.init()
    if torch.distributed.get_rank() == 0:
        if args.log_dir and not args.log_dir.exists():
            args.log_dir.mkdir()
        if args.cache_dir and not args.cache_dir.exists():
            args.cache_dir.mkdir()

    # load error dict
    error_dict = load_error_summary(args.log_dir)

    # block logs except from rank0
    if loggers is None or loggers == {}:
        setup_loggers(args.log_dir, level = logging.INFO)
        for tmp_logger in loggers.values():
            if torch.distributed.get_rank() != 0:
                tmp_logger.setLevel(logging.WARNING)
    for handler in loggers[INFO_FNAME].handlers:
        logger.addHandler(handler)

    # add model name to FxModuleParser log
    fxparser_warning_path = args.log_dir / FXMODULE_PARSER_WARNING_FNAME
    file_handler = logging.FileHandler(fxparser_warning_path)
    from  nnscaler.graph.parser.parser import _logger
    _logger.addHandler(file_handler)
    _logger.warning(f"\n{args.model_name}")

    # Instantiate and use the compiler
    compiler = HFCompiler(args)
    compiler.compile_hf_worker()

