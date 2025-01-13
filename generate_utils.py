import torch
from typing import List, Set, Tuple, Callable
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import generate
class WhisperGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.enc_dec_model

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""

        if maxlen > self.model.cfg.decoder_seq_length - 1:
            maxlen = self.model.cfg.decoder_seq_length - 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
        audio: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            self.encoder_hidden_states = self.model.encode(*audio, reconfigure_microbatch=False)

        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)


        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = tokens.new_zeros(micro_batch_size, tokens2use.size(-1))

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
        batch = [self.encoder_hidden_states, audio[1], tokens2use, attention_mask_repeat, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

def megatron_whisper_generate(model, inputs, tokenizer, inference_params, audio, **strategy_args):
    # reproduce the old compute_prob method
    # a very special case
    if inference_params['compute_logprob']:
        # need to overwrite some configuration, make it immutable
        inference_params = inference_params.copy()
        inference_params = inference_params.copy()
        inference_params['max_length'] = 1
        inference_params['all_probs'] = True
        inference_params["add_BOS"] = False
        inference_params['greedy'] = True
        response = generate(
            model,
            inputs=inputs,
            tokens_to_generate=inference_params['max_length'],
            all_probs=inference_params['all_probs'],
            compute_logprob=inference_params['compute_logprob'],
            temperature=inference_params['temperature'],
            add_BOS=inference_params['add_BOS'],
            top_k=inference_params['top_k'],
            top_p=inference_params['top_p'],
            greedy=inference_params['use_greedy'],
            repetition_penalty=inference_params['repetition_penalty'],
            end_strings=inference_params['end_strings'],
            min_tokens_to_generate=inference_params['min_length'],
            compute_attention_mask=inference_params.get("compute_attention_mask", True),
            image_list=audio,
            **strategy_args,
        )
        compute_prob_response = get_computeprob_response(tokenizer, response, inputs)
        return compute_prob_response

    if not isinstance(inputs, (list, tuple)):
        raise NotImplementedError(f"unknown type {type(inputs)} is not implemented")

    output = generate(
        model,
        inputs=inputs,
        tokens_to_generate=inference_params['max_length'],
        all_probs=inference_params['all_probs'],
        compute_logprob=inference_params['compute_logprob'],
        temperature=inference_params['temperature'],
        add_BOS=inference_params['add_BOS'],
        top_k=inference_params['top_k'],
        top_p=inference_params['top_p'],
        greedy=inference_params['use_greedy'],
        repetition_penalty=inference_params['repetition_penalty'],
        end_strings=inference_params['end_strings'],
        min_tokens_to_generate=inference_params['min_length'],
        image_list=audio,
        **strategy_args,
    )

    return output