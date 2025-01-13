import os
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from argparse import ArgumentParser
from omegaconf import OmegaConf, open_dict
from lightning.pytorch.trainer.trainer import Trainer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.utils_funcs import load_state_dict_helper, torch_dtype_from_precision
from nemo.utils import logging

from model import MegatronWhisperModel

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        required=True,
        help="Path to huggingface model directory",
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Path to NeMo model and config output directory")
    parser.add_argument(
        "--precision",
        type=str,
        default='bf16',
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--nemo_config_path",
        type=str,
        default="./conf/megatron_whisper_config.yaml",
        help="Precision of output weights."
        "Default Whisper configuration file path",
    )
    args = parser.parse_args()
    return args


def hf_qkv_to_nemo_qkv_weight(param_type, attn_type, q, k, v, head_num, hidden_size):
    if attn_type == 'self':
        old_tensor_shape = q.size()
        new_tensor_shape = (head_num, -1) + old_tensor_shape[1:]
        q = q.view(*new_tensor_shape)
        k = k.view(*new_tensor_shape)
        v = v.view(*new_tensor_shape)
        qkv = torch.empty((0, hidden_size // head_num) + old_tensor_shape[1:])
        for i in range(head_num):
            qkv = torch.cat((qkv, q[i: i + 1], k[i : i + 1], v[i : i + 1]))
        if param_type == 'weight':
            qkv = qkv.reshape(-1, hidden_size)
        else:
            qkv = qkv.reshape(-1)
        return qkv

    else:
        old_tensor_shape = k.size()
        new_kv_tensor_shape = (head_num, -1) + old_tensor_shape[1:]
        k = k.view(*new_kv_tensor_shape)
        v = v.view(*new_kv_tensor_shape)
        kv = torch.empty((0, hidden_size // head_num) + old_tensor_shape[1:])
        for i in range(head_num):
            kv = torch.cat((kv, k[i : i + 1], v[i : i + 1]))
        if param_type == 'weight':
            kv = kv.reshape(-1, hidden_size)
        else:
            kv = kv.reshape(-1)
        return q, kv

def load_config(nemo_config_dir, nemo_config_name, hf_config, hf_model):
    with initialize_config_dir(version_base=None, config_dir=nemo_config_dir):
        nemo_config = compose(config_name=nemo_config_name)
    OmegaConf.set_struct(nemo_config, True)
    nemo_config.model.mcore_t5 = True
    nemo_config.model.tokenizer.type = nemo_config.model.feature_extractor.path = hf_config._name_or_path
    nemo_config.model.encoder_seq_length = hf_config.max_source_positions
    nemo_config.model.decoder_seq_length = hf_config.max_target_positions
    nemo_config.model.encoder.num_layers = hf_config.encoder_layers
    nemo_config.model.decoder.num_layers = hf_config.decoder_layers
    nemo_config.model.encoder.num_attention_heads = nemo_config.model.decoder.num_attention_heads = hf_config.encoder_attention_heads
    nemo_config.model.encoder.activation = nemo_config.model.decoder.activation = hf_config.activation_function
    nemo_config.model.encoder.hidden_size = nemo_config.model.decoder.hidden_size = hf_config.d_model
    nemo_config.model.encoder.ffn_hidden_size = nemo_config.model.decoder.ffn_hidden_size = hf_config.encoder_ffn_dim
    convs = []
    for module in [hf_model.model.encoder.conv1, hf_model.model.encoder.conv2]:
        convs.append([module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0]])
    nemo_config.model.feature_extractor.convs = convs
    return nemo_config

if __name__ == '__main__':
    args = get_args()
    model_id = args.hf_input_path
    output_dir = args.output_dir
    precision = args.precision
    nemo_config = args.nemo_config_path

    os.makedirs(output_dir, exist_ok=True)

    logging.info("loading huggingface model...")
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, low_cpu_mem_usage=True, use_safetensors=True
    )
    nemo_config = Path(nemo_config).resolve()
    hf_config = hf_model.config
    logging.info("loading NeMo config...")
    nemo_config = load_config(str(nemo_config.parent), str(nemo_config.name), hf_config, hf_model)
    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=nemo_config.model.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.model.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'
        plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())
    
    hf_state_dict = hf_model.state_dict()
    num_attn_heads = nemo_config.model.encoder.num_attention_heads
    hidden_size = nemo_config.model.encoder.hidden_size

    nemo_state_dict = {}
    for params in ['weight', 'bias']:
        for i in range(2):
            nemo_state_dict[f'enc_dec_model.convs.{i}.{params}'] = hf_state_dict[f'model.encoder.conv{i+1}.{params}']

    nemo_state_dict['enc_dec_model.encoder_position.weight'] = hf_state_dict['model.encoder.embed_positions.weight']
    word_embeddings = hf_state_dict['model.decoder.embed_tokens.weight']
    if word_embeddings.size(0) % nemo_config.model.make_vocab_size_divisible_by != 0:
        vocab_size = (word_embeddings.size(0) // nemo_config.model.make_vocab_size_divisible_by + 1) * nemo_config.model.make_vocab_size_divisible_by
        new_word_embeddings = word_embeddings.new_zeros((vocab_size, word_embeddings.size(1)))
        new_word_embeddings[:word_embeddings.size(0)] = word_embeddings
    else:
        new_word_embeddings = word_embeddings
    nemo_state_dict['enc_dec_model.embedding.word_embeddings.weight'] = new_word_embeddings
    nemo_state_dict['enc_dec_model.embedding.position_embeddings.weight'] = hf_state_dict['model.decoder.embed_positions.weight']

    for module in ["encoder", "decoder"]:
        logging.info(f"Converting the parameters in {module}...")
        for params in ['weight', 'bias']:
            for i in range(hf_config.num_hidden_layers):
                nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.input_layernorm.{params}'] = hf_state_dict[f'model.{module}.layers.{i}.self_attn_layer_norm.{params}']
                self_q_params = hf_state_dict[f'model.{module}.layers.{i}.self_attn.q_proj.{params}']
                self_k_params = hf_state_dict[f'model.{module}.layers.{i}.self_attn.k_proj.{params}'] if params == 'weight' else self_q_params.new_zeros(self_q_params.size())
                self_v_params = hf_state_dict[f'model.{module}.layers.{i}.self_attn.v_proj.{params}']
                qkv_params = hf_qkv_to_nemo_qkv_weight(
                    param_type=params,
                    attn_type='self',
                    q=self_q_params,
                    k=self_k_params,
                    v=self_v_params,
                    head_num=num_attn_heads,
                    hidden_size=hidden_size
                )
                nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.self_attention.linear_qkv.{params}'] = qkv_params

                nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.self_attention.linear_proj.{params}'] = hf_state_dict[f'model.{module}.layers.{i}.self_attn.out_proj.{params}']

                if module == "decoder":
                    nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.pre_cross_attn_layernorm.{params}'] = hf_state_dict[f'model.decoder.layers.{i}.encoder_attn_layer_norm.{params}']
                    cross_q_params = hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.q_proj.{params}']

                    cross_k_params = hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.k_proj.{params}'] if params == 'weight' else cross_q_params.new_zeros(cross_q_params.size())
                    cross_v_params = hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.v_proj.{params}']
                    q_params, kv_params = hf_qkv_to_nemo_qkv_weight(
                        param_type=params,
                        attn_type='cross',
                        q=cross_q_params,
                        k=cross_k_params,
                        v=cross_v_params,
                        head_num=num_attn_heads,
                        hidden_size=hidden_size
                    )
                    nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_q.{params}'] = q_params
                    nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_kv.{params}'] = kv_params
                    
                    nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_proj.{params}'] = hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.out_proj.{params}']

                nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.pre_mlp_layernorm.{params}'] = hf_state_dict[f'model.{module}.layers.{i}.final_layer_norm.{params}']

                for j in range(1, 3):
                    nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.mlp.linear_fc{j}.{params}'] = hf_state_dict[f'model.{module}.layers.{i}.fc{j}.{params}']
            
            nemo_state_dict[f'enc_dec_model.{module}.final_layernorm.{params}'] = hf_state_dict[f'model.{module}.layer_norm.{params}']

    keys = list(nemo_state_dict.keys())
    for key in keys:
        nemo_state_dict[key.replace('enc_dec_model.', 'enc_dec_model.module.', 1)] = nemo_state_dict.pop(key)
    nemo_model = load_state_dict_helper(MegatronWhisperModel, nemo_config.model, trainer, nemo_state_dict)
    nemo_model = nemo_model.to(dtype=torch_dtype_from_precision(precision))
    
    nemo_model.cfg.use_cpu_initialization = False

    logging.info(f"Saving checkpoint and configuation file...")
    nemo_model.save_to(f'{output_dir}/model.nemo')
    del nemo_config.model.precision
    OmegaConf.save(nemo_config, f'{output_dir}/config.yaml')