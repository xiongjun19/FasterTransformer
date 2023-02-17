
'''
Convert huggingface GPT model. Use https://huggingface.co/gpt2 as demo.
'''

import argparse
import configparser
import multiprocessing
import numpy as np
from pathlib import Path
import torch
from typing import Optional
from typing import Union

import os
import sys
from transformers import GPT2Model # transformers-4.10.0-py3
from transformers.configuration_utils import PretrainedConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def split_and_convert_process(i, saved_dir,factor,key,args, val):

    if key.find("input_layernorm.weight") != -1 or key.find("input_layernorm.bias") != -1 or \
        key.find("attention.dense.bias") != -1 or key.find("post_attention_layernorm.weight") != -1 or \
        key.find("post_attention_layernorm.bias") != -1 or key.find("mlp.dense_4h_to_h.bias") != -1 or \
        key.find("final_layernorm.weight") != -1 or key.find("final_layernorm.bias") != -1:

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir + "/model." + key + ".bin"
            val.tofile(saved_path)

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:

        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        local_dim = (int)(val.shape[-1] / 3)

        val = val.reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        hidden_dim = val.shape[0]
        local_dim = (int)(val.shape[-1] / 3)

        val = val.reshape(hidden_dim, 3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    else:
        print("[ERROR] cannot find key '{}'".format(key))


def get_config(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # if _is_offline_mode and not local_files_only:
        #     logger.info("Offline mode: forcing local_files_only=True")
        #     local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs
        return config, model_args, model_kwargs


def load_gpt2_model(in_file):
    model_cfg, model_args, model_kwargs = get_config(GPT2Model, in_file)
    org_model = GPT2Model(model_cfg, *model_args, **model_kwargs)
    org_model = org_model.half()
    return org_model


def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)
    ckpt_name = args.in_file

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = GPT2Model.from_pretrained(args.in_file).to(torch_device)
    model = load_gpt2_model(args.in_file)

    hf_config = vars(model.config)

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    config["gpt"] = {}
    try:
        config["gpt"]["model_name"] = "gpt" if hf_config["_name_or_path"] == '' else hf_config["_name_or_path"]
        config["gpt"]["head_num"] = str(hf_config["n_head"])
        n_embd = hf_config["n_embd"]
        config["gpt"]["size_per_head"] = str(n_embd // hf_config["n_head"])
        config["gpt"]["inter_size"] = str(n_embd * 4)
        config['gpt']['max_pos_seq_len'] = str(hf_config['n_positions'])
        config["gpt"]["num_layer"] = str(hf_config["n_layer"])
        config["gpt"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gpt"]["start_id"] = str(hf_config["bos_token_id"])
        config["gpt"]["end_id"] = str(hf_config["eos_token_id"])
        config['gpt']['weight_data_type'] = args.weight_data_type
        config["gpt"]["tensor_para_size"] = str(args.infer_gpu_num)
        with open(saved_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
    except:
        print(f"Fail to save the config in config.ini.")

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    huggingface_model_name_pattern = [
        "ln_1.bias",
        "ln_1.weight",
        "attn.c_attn.bias",
        "attn.c_attn.weight",
        "attn.c_proj.bias",
        "attn.c_proj.weight",
        "ln_2.bias",
        "ln_2.weight",
        "mlp.c_fc.bias",
        "mlp.c_fc.weight",
        "mlp.c_proj.bias",
        "mlp.c_proj.weight",
    ]
    
    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    pool = multiprocessing.Pool(args.processes)
    for name, param in model.named_parameters():
        if name.find("weight") == -1 and name.find("bias") == -1:
            continue
        if name == 'wpe.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.wpe.bin")
        elif name == 'wte.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.wte.bin")
        elif name == 'ln_f.bias':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.final_layernorm.bias.bin")
        elif name == 'ln_f.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.final_layernorm.weight.bin")
        elif name == 'lm_head.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.lm_head.weight.bin")
        else:
            for i in range(len(huggingface_model_name_pattern)):
                if name.find(huggingface_model_name_pattern[i]) != -1:
                    new_name = name.replace("h.", "layers.").replace(huggingface_model_name_pattern[i], ft_model_name_pattern[i])
                    pool.starmap(split_and_convert_process,
                                [(0, saved_dir, factor, new_name, args,
                                    param.detach().cpu().numpy().astype(np_weight_data_type))], )

    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)", default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    split_and_convert(args)
