import torch
from hydra import compose, initialize_config_dir
from argparse import ArgumentParser
from omegaconf import OmegaConf, open_dict
from lightning.pytorch.trainer.trainer import Trainer
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, pipeline, AutoConfig

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
    parser.add_argument(
        "--nemo_input_path",
        type=str,
        default=None,
        required=True,
        help="Path to huggingface model directory",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to NeMo output path")
    parser.add_argument(
        "--precision",
        type=str,
        default='bf16',
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    args = parser.parse_args()
    return args


def convert(nemo_model, hf_model, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    
    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback
    logging.info(f"Using precision {dtype}")

    param_to_weights = lambda param: param.to(dtype)
    hf_state_dict = dict()
    nemo_state_dict = nemo_model.state_dict()
    hidden_size = nemo_model.cfg.hidden_size
    head_num = nemo_model.cfg.num_attention_heads
    num_layers = nemo_model.cfg.num_layers
    ffn_hidden_size = nemo_model.cfg.ffn_hidden_size
    for params in ['weight', 'bias']:
        for i in range(2):
            hf_state_dict[f'model.encoder.conv{i+1}.{params}'] = nemo_state_dict[f'enc_dec_model.convs.{i}.{params}']

    hf_state_dict['model.encoder.embed_positions.weight'] = nemo_state_dict['enc_dec_model.encoder_position.weight']
    word_embeddings = nemo_state_dict['enc_dec_model.embedding.word_embeddings.weight']
    hf_state_dict['model.decoder.embed_tokens.weight'] = word_embeddings[:hf_model.config.vocab_size]
    hf_state_dict['model.decoder.embed_positions.weight'] = nemo_state_dict['enc_dec_model.embedding.position_embeddings.weight']

    for module in ["encoder", "decoder"]:
        logging.info(f"Converting the parameters in {module}...")
        for params in ['weight', 'bias']:
            for i in range(hf_config.num_hidden_layers):
                hf_state_dict[f'model.{module}.layers.{i}.self_attn_layer_norm.{params}'] = nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.input_layernorm.{params}']

                qkv_params = nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.self_attention.linear_qkv.{params}']
                q_params, k_params, v_params = nemo_qkv_to_hf_qkv_weight(
                    param_type=params,
                    attn_type='self',
                    qkv=qkv_params,
                    head_num=head_num,
                    hidden_size=hidden_size
                )
                hf_state_dict[f'model.{module}.layers.{i}.self_attn.q_proj.{params}'] = q_params
                if params != 'bias':
                    hf_state_dict[f'model.{module}.layers.{i}.self_attn.k_proj.{params}'] = k_params
                else:
                    assert (k_params == 0).all(), "key bias is not zero"
                hf_state_dict[f'model.{module}.layers.{i}.self_attn.v_proj.{params}'] = v_params

                hf_state_dict[f'model.{module}.layers.{i}.self_attn.out_proj.{params}'] = nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.self_attention.linear_proj.{params}']

                if module == "decoder":
                    hf_state_dict[f'model.decoder.layers.{i}.encoder_attn_layer_norm.{params}'] = nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.pre_cross_attn_layernorm.{params}']

                    q_params, k_params, v_params = nemo_qkv_to_hf_qkv_weight(
                        param_type=params,
                        attn_type='cross',
                        q=nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_q.{params}'],
                        kv=nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_kv.{params}'],
                        head_num=head_num,
                        hidden_size=hidden_size
                    )
                    hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.q_proj.{params}'] = q_params
                    if params != 'bias':
                        hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.k_proj.{params}'] = k_params
                    else:
                        assert (k_params == 0).all(), "key bias is not zero"
                    hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.v_proj.{params}'] =         v_params         
                    hf_state_dict[f'model.decoder.layers.{i}.encoder_attn.out_proj.{params}'] = nemo_state_dict[f'enc_dec_model.decoder.layers.{i}.cross_attention.linear_proj.{params}']

                hf_state_dict[f'model.{module}.layers.{i}.final_layer_norm.{params}'] = nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.pre_mlp_layernorm.{params}']

                for j in range(1, 3):
                    hf_state_dict[f'model.{module}.layers.{i}.fc{j}.{params}'] = nemo_state_dict[f'enc_dec_model.{module}.layers.{i}.mlp.linear_fc{j}.{params}']
            
            hf_state_dict[f'model.{module}.layer_norm.{params}'] = nemo_state_dict[f'enc_dec_model.{module}.final_layernorm.{params}']
    hf_state_dict = {k: param_to_weights(v) for k, v in hf_state_dict.items()}
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    assert len(unexpected_keys) == 0 and missing_keys[0] == 'proj_out.weight' and len(missing_keys) == 1, "state dict not matched"

    return hf_model
    
def nemo_qkv_to_hf_qkv_weight(param_type, attn_type, head_num, hidden_size, qkv=None, q=None, kv=None):
    if attn_type == 'self':
        assert qkv is not None, "qkv matrix should not be none for self attention"
        qkv_dim = head_num * 3
        qkv = qkv.reshape(qkv_dim, hidden_size // head_num, -1)
        q = qkv[torch.arange(0, qkv_dim, 3)]
        k = qkv[torch.arange(1, qkv_dim, 3)]
        v = qkv[torch.arange(2, qkv_dim, 3)]
    else:
        assert q is not None and kv is not None, "q and kv matrix should not be none for cross attention"
        kv_dim = head_num * 2
        kv = kv.reshape(kv_dim, hidden_size // head_num, -1)
        k = kv[torch.arange(0, kv_dim, 2)]
        v = kv[torch.arange(1, kv_dim, 2)]

    if param_type == 'weight':
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
    else:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    return q, k, v

def load_config(nemo_ckpt, hf_dir, cpu_only=False):
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    nemo_config =  MegatronWhisperModel.restore_from(nemo_ckpt, trainer=dummy_trainer, return_config=True)
    nemo_config.tensor_model_parallel_size = 1
    nemo_config.pipeline_model_parallel_size = 1
    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None
    
    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    nemo_model = MegatronWhisperModel.restore_from(
        nemo_ckpt, trainer=dummy_trainer, override_config_path=nemo_config, map_location=map_location
    )

    hf_config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    feature_extractor = AutoProcessor.from_pretrained(hf_dir)

    nemo_config.tokenizer.type = nemo_config.feature_extractor.path = hf_config._name_or_path
    hf_config.max_source_positions = nemo_config.encoder_seq_length
    hf_config.max_target_positions = hf_config.max_length = nemo_config.decoder_seq_length
    hf_config.encoder_layers = hf_config.num_hidden_layers = nemo_config.encoder.num_layers
    hf_config.decoder_layers = nemo_config.decoder.num_layers
    
    hf_config.encoder_attention_heads = nemo_config.encoder.num_attention_heads
    hf_config.decoder_attention_heads = nemo_config.decoder.num_attention_heads
    hf_config.activation_function = nemo_config.decoder.activation
    hf_config.d_model = nemo_config.decoder.hidden_size
    hf_config.encoder_ffn_dim = nemo_config.encoder.ffn_hidden_size
    hf_config.decoder_ffn_dim = nemo_config.decoder.ffn_hidden_size
    hf_config.vocab_size = nemo_model.enc_dec_model.module.embedding.word_embeddings.weight.size(0)

    hf_model = AutoModelForSpeechSeq2Seq.from_config(hf_config)
    return nemo_model, hf_config, hf_model, tokenizer, feature_extractor
    
if __name__ == '__main__':
    args = get_args()
    hf_dir = args.hf_input_path
    nemo_ckpt = args.nemo_input_path
    precision = args.precision
    output_dir = args.output_path
    logging.info("loading huggingface model...")
    nemo_model, hf_config, hf_model, tokenizer, feature_extractor = load_config(nemo_ckpt, hf_dir)
    hf_model = convert(nemo_model, hf_model, precision)
    logging.info(f"Saving converted models...")
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    
