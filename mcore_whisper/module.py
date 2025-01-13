# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.models.T5.t5_model import T5Model, t5_position_ids
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

def get_positional_embedding(hidden_size, length):
    if hidden_size % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(hidden_size))
    pe = torch.zeros(length, hidden_size)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, hidden_size, 2, dtype=torch.float) *
                         -(math.log(10000.0) / hidden_size)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    positional_embedding = nn.Embedding.from_pretrained(pe, freeze=False)
     
    return positional_embedding

class WhisperModel(T5Model):
    def __init__(
        self,
        config,
        encoder_config: TransformerConfig,
        transformer_encoder_layer_spec: ModuleSpec,
        transformer_decoder_layer_spec: ModuleSpec,
        convs,
        vocab_size: int,
        encoder_max_length: int,
        decoder_max_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        encoder_position_embedding_type: str = "sinusoidal",
        decoder_position_embedding_type: str = "learned_absolute",
        add_encoder: bool =True,
        add_decoder: bool = True,
    ):
        super(T5Model, self).__init__(config=config)
        self.config: TransformerConfig = config
        self.encoder_config: TransformerConfig = encoder_config
        self.transformer_encoder_layer_spec: ModuleSpec = transformer_encoder_layer_spec
        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        self.vocab_size = vocab_size
        self.max_encoder_sequence_length = encoder_max_length
        self.max_decoder_sequence_length = decoder_max_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_position_embedding_type = encoder_position_embedding_type
        self.position_embedding_type = decoder_position_embedding_type
        self.encoder_hidden_state = None

        self.model_type = ModelType.encoder_and_decoder

        # Tells schedules.py that this model has a skip connection
        # between the encoder's output and the decoder
        # (and hence both the encoder and decoder's tensors are required for correct backprop).
        self.xattn_needed = True

        # specify the position embeddings as a member
        # variable in the T5 class so that they are easy to
        # find for `finalize_model_grads._allreduce_position_embedding_grads`
        # self.position_embeddings = None
        if self.pre_process:
            if self.add_encoder:
                self.convs = nn.ModuleList([
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
                    for c_in, c_out, kernel_size, stride, padding in convs
                ])
                self.activation_func = self.encoder_config.activation_func
                self.encoder_position = get_positional_embedding(self.encoder_config.hidden_size, self.max_encoder_sequence_length)

            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_decoder_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )
            # if decoder_position_embedding_type == "learned_absolute":
            #     self.position_embeddings = self.embedding.position_embeddings
            # else:
            #     self.position_embeddings = None

        # Rotary Position Embeddings
        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # Transformer encoder
        encoder_spec, decoder_spec = (
            self.transformer_encoder_layer_spec,
            self.transformer_decoder_layer_spec,
        )
        if self.add_encoder:
            self.encoder = TransformerBlock(
                config=self.encoder_config,
                spec=encoder_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        else:
            self.encoder = None

        if self.add_decoder:
            # Transformer decoder
            self.decoder = TransformerBlock(
                config=self.config,
                spec=decoder_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        else:
            self.decoder = None

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=True,
                gather_output=not parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()


    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:

        sharded_sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.pre_process and self.add_encoder:
            encoder_position_prefix = f'{prefix}encoder_position.'
            sharded_sd.update(sharded_state_dict_default(self.encoder_position, encoder_position_prefix, sharded_offsets, metadata))
            conv_prefix = f'{prefix}convs.'
            sharded_sd.update(sharded_state_dict_default(self.convs, conv_prefix, sharded_offsets, metadata))
        return sharded_sd

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attn_mask: Tensor,
        decoder_attn_mask: Tensor,
        encoder_decoder_attn_mask: Tensor,
        lm_labels: Tensor = None,
        encoder_hidden_states: Tensor = None,
        output_encoder_hidden_only: bool = False,
        inference_params: InferenceParams = None,
    ):
        ## Encoder forward
        if encoder_hidden_states is None:
            # Encoder embedding.
            if self.pre_process:
                for conv in self.convs:
                    encoder_input_ids = self.activation_func(conv(encoder_input_ids))
                # Encoder position ids
                encoder_input = encoder_input_ids.transpose(1, 2)
                encoder_position_ids = whisper_position_ids(encoder_input)
                encoder_input += self.encoder_position(encoder_position_ids)
            else:
                # intermediate stage of pipeline
                encoder_input = None

            # Run encoder.
            encoder_input = encoder_input.transpose(0, 1).contiguous()    # batch-first to seq-first

            encoder_hidden_states = self.encoder(
                hidden_states=encoder_input,
                attention_mask=encoder_attn_mask,
                inference_params=inference_params,
            )
        # Return encoder hiddenstates if output_encoder_hidden_only is True
        if output_encoder_hidden_only:
            return encoder_hidden_states

        ## Decoder forward
        # Decoder position ids
        decoder_position_ids = t5_position_ids(decoder_input_ids)
        if inference_params:
            decoder_position_ids = decoder_position_ids + inference_params.sequence_len_offset

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(
                input_ids=decoder_input_ids, position_ids=decoder_position_ids
            )
        else:
            # intermediate stage of pipeline
            decoder_input = None  ### should it take encoder_hidden_states

        # Run decoder.
        decoder_hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=decoder_attn_mask,
            context=encoder_hidden_states,
            context_mask=encoder_decoder_attn_mask,
            inference_params=inference_params,
        )

        # Return if not post_process
        if not self.post_process:
            return decoder_hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(decoder_hidden_states, weight=output_weight)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(lm_labels, logits)

        return loss

def whisper_position_ids(token_ids: Tensor) -> Tensor:
    """Calculate position ids from token ids
    Args:
        token_ids (Tensor): input tokens

    Returns:
        Tensor: position ids
    """
    b, seq_length, _ = token_ids.size()
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).repeat(b, 1)

    return position_ids