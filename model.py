from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from dataclasses import fields
from typing import Any, Dict, List, Optional
import itertools
import copy
import json
from jiwer import wer, cer

import torch
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import get_datasets_weights_and_num_samples
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import AttnMaskType
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.utils import AppState, logging
from nemo.collections.nlp.parts.utils_funcs import activation_to_func
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.modules.common.megatron.utils import build_attention_mask_3d
from nemo.collections.nlp.modules.common.text_generation_utils import (
    compute_beam_search_len_penalty,
    get_sampling_token_fn,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from megatron.core import InferenceParams, parallel_state
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.num_microbatches_calculator import (
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
        reconfigure_num_microbatches_calculator,
    )
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

from mcore_whisper.module import WhisperModel as MCoreWhisperModel
from mcore_whisper.whisper_spec import (
    get_whisper_encoder_with_local_block_spec,
    get_whisper_encoder_with_transformer_engine_block_spec,
    get_whisper_decoder_with_local_block_spec,
    get_whisper_decoder_with_transformer_engine_block_spec,
)
from dataset import WhisperDataset
from generate_utils import WhisperGenerationStrategy, megatron_whisper_generate
class MegatronWhisperModel(MegatronLMEncoderDecoderModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.set_inference_config(
            {
                'max_length': self.cfg.data.validation_ds.get('max_dec_length', 1),
                'use_greedy': self.cfg.data.validation_ds.get('greedy', True),
                'top_k': self.cfg.data.validation_ds.get('top_k', 20),
                'top_p': self.cfg.data.validation_ds.get('top_p', 0.9),
                'temperature': self.cfg.data.validation_ds.get('temperature', 1),
                'compute_logprob': self.cfg.data.validation_ds.get('compute_logprob', False),
                'add_BOS': self.cfg.data.validation_ds.get('add_BOS', False),
                'repetition_penalty': self.cfg.data.validation_ds.get('repetition_penalty', 1),
                'end_strings': self.cfg.data.validation_ds.get('end_strings', [self.tokenizer.eos_token]),
                'all_probs': self.cfg.data.validation_ds.get('all_probs', False),
                'min_length': self.cfg.data.validation_ds.get('min_dec_length', 1),
            }
        )
        self.inference_params = None
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.cfg.feature_extractor.path)

    def setup_optimizer_param_groups(self):
        super().setup_optimizer_param_groups()
        for group in self._optimizer_param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config
    @property
    def model_name(self):
        return "Whisper"

    @property
    def max_decoder_seq_length(self) -> int:
        return self.cfg.decoder_seq_length

    @property
    def max_encoder_seq_length(self) -> int:
        return self.cfg.encoder_seq_length

    def get_feature(self, x):
        return self.feature_extractor(
            raw_speech=x,
            pad_to_multiple_of=8,
            return_tensors='pt',
            return_attention_mask=True,
            padding=True,
            do_normalize=True,
            device="cuda",
        )

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        if self.cfg.get('transformer_engine', False):
            enc_dec_spec_fns = (
                get_whisper_encoder_with_transformer_engine_block_spec,
                get_whisper_decoder_with_transformer_engine_block_spec,
            )
        else:
            enc_dec_spec_fns = (
                get_whisper_encoder_with_local_block_spec,
                get_whisper_decoder_with_local_block_spec,
            )

        en_block_spec = enc_dec_spec_fns[0](self.cfg.encoder.num_layers)
        de_block_spec = enc_dec_spec_fns[1](self.cfg.decoder.num_layers)
        encoder_config = copy.deepcopy(self.transformer_config)
        encoder_config.num_layers = self.cfg.encoder.num_layers
        encoder_config.position_embedding_type = self.cfg.encoder.position_embedding_type
        if self.cfg.pipeline_model_parallel_size > 1:
                assert (
                    self.cfg.pipeline_model_parallel_split_rank is not None
                ), "Need to know how to shard the encoder & decoder."
                encoder_config.pipeline_model_parallel_size = self.cfg.pipeline_model_parallel_split_rank

        # get model parallel configs from the base class
        model = MCoreWhisperModel(
            config=self.transformer_config,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=de_block_spec,
            convs=self.cfg.feature_extractor.convs,
            vocab_size=self.padded_vocab_size,
            encoder_max_length=self.cfg.encoder_seq_length,
            decoder_max_length=self.cfg.decoder_seq_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            parallel_output=True,
            share_embeddings_and_output_weights=self.cfg.get('share_decoder_tokens_head_embeddings', False),
            encoder_position_embedding_type=self.cfg.encoder.get('position_embedding_type', 'learned_absolute'),
            decoder_position_embedding_type=self.cfg.decoder.get('position_embedding_type', 'learned_absolute'),
            add_encoder=add_encoder,
            add_decoder=add_decoder
        )
        self.freeze_model(model)
        return model

    def freeze_model(self, model):
        for name, param in model.named_parameters():
            if (self.cfg.freeze_encoder and (name.startswith('encoder.') or name.startswith('encoder_position.'))) or \
            (self.cfg.encoder.get('position_embedding_type', 'sinusoidal') == 'sinusoidal' and name.startswith('encoder_position.')) or \
            (self.cfg.freeze_qkv_bias and (name.endswith('kv.bias') or name.endswith('q.bias'))):
                param.requires_grad = False

    def _build_dataset(self, data_cfg, is_train=True):
        if is_train:
            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)
        # pad_seq_length_to_mult = (
        #     8 * self.cfg.get('tensor_model_parallel_size', 1) if self.cfg.get('sequence_parallel', False) else 16
        # )
        # pad_seq_length_to_mult *= self.cfg.get('context_parallel_size', 1)
        downsample_rate = 1
        for conv in  self.cfg.feature_extractor.convs:
            downsample_rate *= conv[3]
        datasets = []
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = WhisperDataset(
                file_path=file_path,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                sample_rate=data_cfg.sample_rate,
                downsample_rate=downsample_rate,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                audio_key=data_cfg.audio_key,
                text_key=data_cfg.text_key,
                task_key=data_cfg.task_key,
                lang_key=data_cfg.lang_key,
                prompt_key=data_cfg.prompt_key,
                truncation_field=data_cfg.get('truncation_field', 'prompt'),
                truncation_method=data_cfg.get('truncation_method', 'left'),
                encoder_padding_method=data_cfg.get('encoder_padding_method', 'max_length'),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                memmap_workers=data_cfg.get('memmap_workers', None),
                prompt_template=data_cfg.prompt_template,
                global_sample_mapping=data_cfg.get('global_sample_mapping', False),
                is_test=not is_train,
            )
            datasets.append(dataset)
        if is_train:
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def build_train_valid_test_datasets(self, stage='train'):
        if stage != 'test':
            logging.info('Building Whisper validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds, is_train=False)
            if self._validation_ds:
                logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')
        else:
            self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)
            logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building Whisper traing datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

    def build_data_loader(self, dataset, data_cfg, consumed_samples, num_workers):
        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=data_cfg.drop_last,
            pad_samples_to_global_batch_size=not data_cfg.drop_last,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=True if num_workers > 0 else False,
        )


    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                self._train_ds, self.cfg.data.train_ds, consumed_samples, self.cfg.data.train_ds.num_workers,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = [self.build_data_loader(
                dataset, self.cfg.data.validation_ds, consumed_samples, self.cfg.data.validation_ds.num_workers,
            ) for dataset in self._validation_ds]

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = [self.build_data_loader(dataset, self.cfg.data.test_ds, consumed_samples, 0) for dataset in self._test_ds]

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'state_dict' in checkpoint and checkpoint['state_dict']:
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    checkpoint_state_dict = checkpoint['state_dict'][f'model_{index}']
                else:
                    checkpoint_state_dict = checkpoint['state_dict']
                # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                checkpoint_state_dict = {
                    key.replace('model.', ''): checkpoint_state_dict.pop(key)
                    for key in list(checkpoint_state_dict.keys())
                }
                module.load_state_dict(checkpoint_state_dict, strict=True)
        else:
            checkpoint['state_dict'] = {}

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            # convert to list if not already converted.
            if isinstance(batch, dict):
                # convert to list if not already converted.
                batch = self._process_batch(batch)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]
            (
                encoder_input_ids,
                decoder_input_ids,
                loss_mask,
                lm_labels,
                encoder_attn_mask,
                decoder_attn_mask,
                batch_data,
            ) = batch
            encoder_attn_mask = encoder_attn_mask < 0.5
            decoder_attn_mask = decoder_attn_mask < 0.5
            if self.cfg.get('transformer_engine', False):
                encoder_attn_mask_3d = encoder_attn_mask.unsqueeze(1).unsqueeze(1)
                decoder_attn_mask_3d = decoder_attn_mask.unsqueeze(1).unsqueeze(1)
                enc_dec_attn_mask_3d = (
                    decoder_attn_mask_3d, 
                    encoder_attn_mask_3d,
                )
            else:
                encoder_attn_mask_3d = build_attention_mask_3d(encoder_attn_mask, encoder_attn_mask, AttnMaskType.padding).unsqueeze(1)
                decoder_attn_mask_3d = build_attention_mask_3d(decoder_attn_mask, decoder_attn_mask, AttnMaskType.causal).unsqueeze(1)
                enc_dec_attn_mask_3d = build_attention_mask_3d(decoder_attn_mask, encoder_attn_mask, AttnMaskType.padding).unsqueeze(1)

            output = model(  # model is MCoreT5Model
                encoder_input_ids,  # encoder_input_ids
                decoder_input_ids,  # decoder_input_ids
                encoder_attn_mask_3d,  # encoder_attn_mask
                decoder_attn_mask_3d,  # decoder_attn_mask
                enc_dec_attn_mask_3d,  # encoder_decoder_attn_mask
                lm_labels,  # lm_labels
            )

            def loss_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations
                    loss_dict = output_tensor
                    output_tensor = loss_dict.pop("output")
                    # compute reconstruction (tokens) only loss from per-token reconstruction loss
                    tokens_loss = self.loss_func(loss_mask, output_tensor)
                    loss_dict["tokens_loss"] = tokens_loss
                    tokens_loss_weight = loss_dict.get("tokens_loss_weight", 1.0)
                    # compute total loss
                    loss = loss_dict["loss"] = loss_dict["hiddens_loss"] + tokens_loss_weight * tokens_loss
                    # average losses across data parallel group
                    loss_dict = {
                        k: average_losses_across_data_parallel_group([v.mean()]) for k, v in loss_dict.items()
                    }
                else:
                    # compute reconstruction (tokens) only loss from per-token reconstruction loss
                    loss = self.loss_func(loss_mask, output_tensor)
                    # average losses across data parallel group
                    reduced_loss = average_losses_across_data_parallel_group([loss])
                    loss_dict = {'loss': reduced_loss}

                return loss, loss_dict

            return output, loss_func

        return fwd_output_and_loss_func

    def on_validation_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.micro_batch_size * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.micro_batch_size * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_validation_epoch_end(self):
        self.on_inference_epoch_end(self.validation_step_outputs, "validation", self.cfg.data.validation_ds)

    def on_test_epoch_end(self):
        self.on_inference_epoch_end(self.test_step_outputs, "test", self.cfg.data.test_ds)

    def gather_and_maybe_write_predictions(self, output, data_cfg, mode, averaged_wer, averaged_cer, dataloader_idx=0):
        # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [
                {'preds': x['sentences'], 'labels': x['labels'], 'prefix': x['prefix']}
                for x in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'prefix': [],
        }
        total_size = 0
        wer_key = f"{mode}_wer_dataloader{dataloader_idx}"
        cer_key = f"{mode}_cer_dataloader{dataloader_idx}"
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for pred, label, prefix in zip(
                    batch['preds'], batch['labels'], batch['prefix']
                ):
                    total_size += 1
                    deduplicated_outputs['preds'].append(pred)
                    deduplicated_outputs['labels'].append(label)
                    deduplicated_outputs['prefix'].append(prefix)

            word_errs = wer(deduplicated_outputs['labels'], deduplicated_outputs['preds'])
            char_errs = cer(deduplicated_outputs['labels'], deduplicated_outputs['preds'])

            self.log(wer_key, word_errs, sync_dist=True)
            self.log(cer_key, char_errs, sync_dist=True)

            averaged_wer.append(word_errs)
            averaged_cer.append(char_errs)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['prefix'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                raise ValueError(
                    f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                )
            self.write_predictions_to_file(
                deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{mode}_dataloader{dataloader_idx}"
            )

        return deduplicated_outputs, total_size

    def write_predictions_to_file(self, outputs, output_file_path_prefix):
        output_file_path = output_file_path_prefix + "_prefix_preds_labels.jsonl"
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['prefix']) == len(outputs['preds']) == len(outputs['labels'])
            )
            for i, p, l in zip(outputs['prefix'], outputs['preds'], outputs['labels']):
                json_string = {'prefix': i, 'pred': p, 'label': l}
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def on_inference_epoch_end(self, outputs, mode, data_cfg):
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        if not outputs or not outputs[0]:
            return
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_wer = []
        averaged_cer = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            # loss = super().on_validation_epoch_end([x['loss'] for x in output])
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            self.log(f'{mode}_loss_dataloader{dataloader_idx}', loss, batch_size=1)
            averaged_loss.append(loss)
            self.gather_and_maybe_write_predictions(output, data_cfg, mode, averaged_wer, averaged_cer, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_wer = sum(averaged_wer) / len(averaged_wer) if len(averaged_wer) >= 1 else 1e3
        averaged_cer = sum(averaged_cer) / len(averaged_cer) if len(averaged_cer) >= 1 else 1e3

        self.log(f"{mode}_loss", averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(f"{mode}_wer", averaged_wer, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(f"{mode}_cer", averaged_cer, prog_bar=True, rank_zero_only=True, batch_size=1)

        logging.info(f'{mode}_loss: {averaged_loss}')
        logging.info(f'{mode}_wer: {averaged_wer}')
        logging.info(f'{mode}_cer: {averaged_cer}')

        return averaged_loss, averaged_wer, averaged_cer

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['text_enc'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
    
    def inference_step(self, dataloader_iter, mode="validation", inference_config=None):
        def process_batch(dataloader_iter):
            batch, batch_idx, dataloader_idx = next(dataloader_iter)
            self._reconfigure_and_process_inference_batch(batch, self.cfg.data.validation_ds if mode == "validation" else self.cfg.data.test_ds)
            text_enc = batch["text_enc"].cuda(non_blocking=True)
            enc_mask = batch["enc_mask"].cuda(non_blocking=True)

            audio = (text_enc, enc_mask)
            prefix = (batch["prefix"].cuda(non_blocking=True), batch["len_arr"].cuda(non_blocking=True))
            raw_prefix = batch["raw_prefix"] if "raw_prefix" in batch else None
            raw_label = batch["raw_label"] if "raw_label" in batch else None
            return batch, batch_idx, dataloader_idx, (prefix, audio, raw_prefix, raw_label)

        loss_batch, batch_idx, dataloader_idx, (prefix, audio, raw_prefix, raw_label) = process_batch(dataloader_iter)
        loss = self.fwd_bwd_step(itertools.chain([loss_batch]), forward_only=True)
        inference_config = self.get_inference_config() if inference_config is None else {}
        if len(inference_config) == 0:
            # set the default inference_config if it is not set.
            inference_config.update(get_default_sampling_params())
            inference_config.update(get_default_length_params())

        strategy_args = {"strategy": WhisperGenerationStrategy(self)}
        outputs = megatron_whisper_generate(self, prefix, self.tokenizer, inference_config, audio, **strategy_args)

        if isinstance(loss, dict):
            outputs.update(loss)
        else:
            outputs['loss'] = loss

        if raw_prefix is not None and raw_label is not None:
            outputs['sentences'] = [hyp[len(p):].strip() for p, hyp in zip(raw_prefix, outputs['sentences'])]
            outputs['labels'] = raw_label
            outputs['prefix'] = [self.tokenizer.tokenizer.decode(p[:l]) for p, l in zip(*prefix)]
        return outputs

    def validation_step(self, dataloader_iter):
        outputs = self.inference_step(dataloader_iter, mode="validation")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_iter.dataloader_idx].append(outputs)
        else:
            self.validation_step_outputs.append(outputs)

    def test_step(self, dataloader_iter):
        outputs = self.inference_step(dataloader_iter, mode="test")
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_iter.dataloader_idx].append(outputs)
        else:
            self.test_step_outputs.append(outputs)

    def encode(self, tokens_enc, enc_mask, encoder_input=None, batch_data=None, reconfigure_microbatch=True):
        if not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            if reconfigure_microbatch:
                reconfigure_num_microbatches_calculator(
                    rank=0,  # This doesn't matter since it is only used for logging
                    rampup_batch_size=None,
                    global_batch_size=1,
                    micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                    data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
                )

        # If classes that inherit from this class are using a different tokenizer,
        app_state = AppState()
        if tokens_enc is not None:
            global_batch_per_gpu = tokens_enc.size(0)
            encoder_seq_length = tokens_enc.size(1)
        else:
            global_batch_per_gpu = encoder_input.size(1)
            encoder_seq_length = encoder_input.size(0)

        num_micro_batches_before_decode = get_num_microbatches()
        # Reconfigure microbatch calculator here to set num microbatches to 1 while decoding since its not clear how to decode with "grad acc".
        # reconfigure back to how things were before encode
        if reconfigure_microbatch:
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        tensor_shape = [encoder_seq_length, global_batch_per_gpu, self.cfg.encoder.hidden_size]

        # build input arguments description
        if tokens_enc is not None:
            batch_for_pipeline = [tokens_enc, enc_mask]
        else:
            if encoder_input is None:
                raise ValueError("At least one of tokens_enc and encoder_input must be provided with not None value")

            batch_for_pipeline = [enc_mask]

        if encoder_input is not None:
            batch_for_pipeline.append(encoder_input)

        forward_step_func = self.get_forward_output_only_func(output_name="hiddens")

        fwd_bwd_func = get_forward_backward_func()

        # Counter intuitively, we need to set decoder_sequence_length=encoder_seq_length
        # because while running `.encode()`, the last hidden states from encoder are passed through
        # as identity through the pipeline.
        # Setting it to anything else will cause hanging due to tensor shape mismatches.
        output_tensor = fwd_bwd_func(
            forward_step_func=forward_step_func,
            data_iterator=iter(
                [
                    batch_for_pipeline,
                ]
            ),
            model=[self.enc_dec_model],
            forward_only=True,
            num_microbatches=1,
            seq_length=encoder_seq_length,
            decoder_seq_length=encoder_seq_length,
            micro_batch_size=get_micro_batch_size(),
        )

        if output_tensor:
            output_tensor = output_tensor[0]['hiddens']
        else:
            output_tensor = torch.zeros(tensor_shape, dtype=self.autocast_dtype).cuda()

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # Broadcast from the last pipeline stage to all other model-parallel ranks.
            torch.distributed.broadcast(
                output_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

        # Reset microbatch calculator to what it was before decoding.
        if reconfigure_microbatch:
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu // num_micro_batches_before_decode,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        # Return the output tensor of encoder and transpose from [seq_len, batch, hidden] to [batch, seq_len, hidden]
        return output_tensor.transpose(1, 0)

    def get_forward_output_only_func(self, output_name="logits", **kwargs):
        """
        args_idx - maps batch into index of args (with None filling gaps)
        output_name - name of output (hiddens for encode, logits for decode)
        kwargs - shared arguments (non tensors)
        """

        def fwd_output_only_func(dataloader_iter, model):
            # Extract batch, batch_idx, dataloader_idx only if dataloader_iter is an object of PTL's _DataFetcherWrapper
            extra_arg = {}
            if isinstance(dataloader_iter, _DataFetcherWrapper):
                batch, _, _ = next(dataloader_iter)
            else:
                batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]
            # when run encoding
            if output_name == "hiddens":
                (
                    encoder_input_ids,
                    encoder_attn_mask,
                ) = batch

                encoder_attn_mask = encoder_attn_mask < 0.5
                if self.cfg.get('transformer_engine', False):
                    encoder_attn_mask_3d = encoder_attn_mask.unsqueeze(1).unsqueeze(1)
                else:
                    encoder_attn_mask_3d = build_attention_mask_3d(encoder_attn_mask, encoder_attn_mask, AttnMaskType.padding).unsqueeze(1)
                output = model(
                    encoder_input_ids,
                    None,
                    encoder_attn_mask_3d,
                    None,
                    None,
                    None,
                    None,
                    output_encoder_hidden_only=True,
                ).contiguous()
            # when run decoding
            elif output_name == "logits":
                (
                    encoder_hidden_states,
                    encoder_attn_mask,
                    decoder_input_ids,
                    decoder_attn_mask,
                    set_inference_key_value_memory,
                    inference_max_sequence_len
                ) = batch

                encoder_attn_mask = encoder_attn_mask < 0.5
                decoder_attn_mask = decoder_attn_mask < 0.5
                if self.cfg.get('transformer_engine', False):
                    encoder_attn_mask_3d = encoder_attn_mask.unsqueeze(1).unsqueeze(1)
                    decoder_attn_mask_3d = decoder_attn_mask.unsqueeze(1).unsqueeze(1)
                    enc_dec_attn_mask_3d = (
                        decoder_attn_mask_3d, 
                        encoder_attn_mask_3d,
                    )
                else:
                    encoder_attn_mask_3d = build_attention_mask_3d(encoder_attn_mask, encoder_attn_mask, AttnMaskType.padding).unsqueeze(1)
                    decoder_attn_mask_3d = build_attention_mask_3d(decoder_attn_mask, decoder_attn_mask, AttnMaskType.causal).unsqueeze(1)
                    enc_dec_attn_mask_3d = build_attention_mask_3d(decoder_attn_mask, encoder_attn_mask, AttnMaskType.padding).unsqueeze(1)

                # re-transpose encoder_hidden_states from [batch, seq_len, hidden] to [seq_len, batch, hidden]
                encoder_hidden_states = encoder_hidden_states.transpose(1, 0)
                if set_inference_key_value_memory[0].item():
                    self.inference_params = InferenceParams(
                        max_batch_size=decoder_input_ids.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = self.inference_params
                output = model(
                    None,
                    decoder_input_ids,
                    encoder_attn_mask_3d,
                    decoder_attn_mask_3d,
                    enc_dec_attn_mask_3d,
                    None,
                    encoder_hidden_states,
                    output_encoder_hidden_only=False,
                    **extra_arg
                ).contiguous()
            else:
                assert output_name in [
                    "hiddens",
                    "logits",
                ], "output_name argument must be either 'hiddens' or 'logits'"

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output.size(1)
                else:
                    self.inference_params.sequence_len_offset += output.size(0)

            def id_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations ("output" is the default output)
                    output_tensor = output_tensor["output"]

                return output_tensor, {output_name: output_tensor}

            return output, id_func

        return fwd_output_only_func

    def _reset_activation_checkpointing_args(self):
        """Disables activation checkpointing completely and saves the values so that
        _restore_activation_checkpointing_args can restore them later. This function must always be
        called before _restore_activation_checkpointing_args.
        """
        # Store values to restore them later.
        self.last_activations_checkpoint_granularity = self.cfg.activations_checkpoint_granularity
        self.last_activations_checkpoint_method = self.cfg.activations_checkpoint_method
        self.last_activations_checkpoint_num_layers = self.cfg.activations_checkpoint_num_layers
        self.last_activations_checkpoint_layers_per_pipeline = self.cfg.activations_checkpoint_layers_per_pipeline

        # Reset config values. Needed for calling generate.
        self.cfg.activations_checkpoint_granularity = None
        self.cfg.activations_checkpoint_method = None
        self.cfg.activations_checkpoint_num_layers = None
        self.cfg.activations_checkpoint_layers_per_pipeline = None

        # Reset model parameters.
        for module in self.get_model_module_list():
            module.encoder.config.recompute_granularity = None
            module.encoder.config.recompute_method = None
            module.encoder.config.recompute_num_layers = None
            module.decoder.config.recompute_granularity = None
            module.decoder.config.recompute_method = None
            module.decoder.config.recompute_num_layers = None

    def _restore_activation_checkpointing_args(self):
        """Restores the activation checkpointing parameters using the values saved by
        _reset_activation_checkpointing_args. This function must never be called before
        _reset_activation_checkpointing_args.
        """
        # Restore config values.
        self.cfg.activations_checkpoint_granularity = self.last_activations_checkpoint_granularity
        self.cfg.activations_checkpoint_method = self.last_activations_checkpoint_method
        self.cfg.activations_checkpoint_num_layers = self.last_activations_checkpoint_num_layers
        self.cfg.activations_checkpoint_layers_per_pipeline = self.last_activations_checkpoint_layers_per_pipeline

        # Restore model parameters.
        for module in self.get_model_module_list():
            module.encoder.config.recompute_granularity = self.last_activations_checkpoint_granularity
            module.encoder.config.recompute_method = self.last_activations_checkpoint_method
            module.encoder.config.recompute_num_layers = self.last_activations_checkpoint_num_layers
            module.decoder.config.recompute_granularity = self.last_activations_checkpoint_granularity
            module.decoder.config.recompute_method = self.last_activations_checkpoint_method
            module.decoder.config.recompute_num_layers = self.last_activations_checkpoint_num_layers

