import json
import re
import torch
import soundfile as sf
from nemo.core.classes import Dataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset, OnlineSampleMapping
from nemo.collections.nlp.modules.common.lm_utils import pad_batch

class WhisperDataset(GPTSFTDataset):
    def __init__(
        self,
        file_path,
        feature_extractor,
        tokenizer,
        max_seq_length=1000,
        min_seq_length=1,
        sample_rate=16000,
        downsample_rate=2,  # downsampling ratio of convolutional layers
        max_num_samples=-1,
        seed=1234,
        audio_key='audio_path',
        text_key='text',
        task_key='task',
        lang_key='lang',
        prompt_key='prompt',
        timestamp_key='timestamp',
        truncation_field='prefix',
        truncation_method='left',
        index_mapping_dir=None,
        memmap_workers=None,
        prompt_template=None,
        global_sample_mapping=False,
        encoder_padding_method='max_length',
        is_test=False,
        sanity_check_dist_workers: bool = True,
    ):
        self.file_path = file_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sr = sample_rate
        self.downsample_rate = downsample_rate
        self.max_seq_length = max_seq_length - 1
        self.min_seq_length = min_seq_length
        self.max_num_samples = max_num_samples
        self.global_sample_mapping = global_sample_mapping
        self.seed = seed
        self.sanity_check_dist_workers = sanity_check_dist_workers

        self.prompt_key = prompt_key
        self.task_key = task_key
        self.lang_key = lang_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.timestamp_key = timestamp_key
        self.prompt_template = prompt_template
        self.prompt_template_keys = re.findall(r'{(.*?)}', self.prompt_template)
        self.encoder_padding_method = encoder_padding_method

        self.truncation_field = truncation_field
        self.truncation_method = truncation_method
        self.is_test = is_test

        self.index_mapping_dir = index_mapping_dir
        self.indexed_dataset = JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=memmap_workers,
            )
        self._build_samples_mapping()

    def batch_manifest(self, batch):
        audio, tokens, token_length, answer_start_idx = [], [], [], []
        max_seq_length, max_prefix_length = 0, 0
        for b in batch:
            audio.append(b['audio'])
            tokens.append(b['token_ids'])
            answer_start_idx.append(b['answer_start_idx'])
            token_length.append(b['token_length'])
            if b['token_length'] > max_seq_length:
                max_seq_length = b['token_length']
            if b['answer_start_idx'] > max_prefix_length:
                max_prefix_length = b["answer_start_idx"]
        return audio, tokens, token_length, answer_start_idx, max_seq_length, max_prefix_length

    def collate_fn(self, batch):
        def refine_enc_length(enc_inputs, enc_attn_masks):
            length = min(enc_inputs.size(-1), enc_attn_masks.size(-1))
            enc_inputs = enc_inputs[..., :length]
            enc_attn_masks = enc_attn_masks[..., :length:self.downsample_rate] * 0
            return enc_inputs, enc_attn_masks

        audio, tokens, token_length, answer_start_idx, max_seq_length, max_prefix_length = self.batch_manifest(batch)
        feats = self.feature_extractor(
            raw_speech=audio, 
            sampling_rate=self.sr, 
            return_tensors="pt",
            return_attention_mask=True,
            padding=self.encoder_padding_method,
            do_normalize=False,
            device="cuda"
        )

        enc_inputs, enc_attn_masks = refine_enc_length(feats.input_features, feats.attention_mask)

        dec_inputs = []
        labels = []
        prefixes = []
        raw_prefixes = []
        raw_labels = []
        loss_masks = torch.zeros(len(batch), max_seq_length)
        dec_attn_masks = torch.ones(len(batch), max_seq_length)
        for i, (token, start_idx, length) in enumerate(zip(tokens, answer_start_idx, token_length)):
            dec_input = token[:-1] + [self.tokenizer.pad_id] * (max_seq_length - length)
            label = token[1:] + [self.tokenizer.pad_id] * (max_seq_length - length)
            prefix = token[: start_idx + 1]
            loss_masks[i, start_idx: length] += 1
            dec_attn_masks[i, :length] *= 0

            raw_prefixes.append(self.tokenizer.ids_to_text(prefix))
            raw_labels.append(self.tokenizer.ids_to_text(token[start_idx:]))
            prefixes.append(prefix)
            dec_inputs.append(dec_input)
            labels.append(label)

        prefixes, prefixes_length_arr = pad_batch(prefixes, self.tokenizer.pad_id, self.max_seq_length - max_prefix_length + 1)
        dec_inputs = torch.LongTensor(dec_inputs)
        labels = torch.LongTensor(labels)
        prefixes = torch.LongTensor(prefixes)
        prefixes_length_arr = torch.LongTensor(prefixes_length_arr)
        if self.is_test:
            return {
                "text_enc": enc_inputs, 
                "text_dec": dec_inputs, 
                "loss_mask": loss_masks, 
                "labels": labels, 
                "raw_prefix": raw_prefixes,
                "raw_label": raw_labels,
                "prefix": prefixes,
                "len_arr": prefixes_length_arr,
                "enc_mask": enc_attn_masks, 
                "dec_mask": dec_attn_masks
            }
        else:
            return {
                "text_enc": enc_inputs, 
                "text_dec": dec_inputs, 
                "loss_mask": loss_masks, 
                "labels": labels, 
                "enc_mask": enc_attn_masks, 
                "dec_mask": dec_attn_masks
            }

    def _process_example(self, example):
        audio, sr = sf.read(example[self.audio_key])
        assert sr == self.sr, "Audio sample rate does not match"

        prompt_template_values = []
        for c in self.prompt_template_keys:
            try:
                prompt_template_values.append(example[c].strip(' '))
            except KeyError as e:
                if c == self.text_key and self.is_test:
                    # allow missing label during testing, if user only wants to do inference without calculating metrics
                    prompt_template_values.append("")
                else:
                    raise e
        template_strings, template_strings_keys = self._separate_template(prompt_template_values)
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        total_length = sum([len(ids) for ids in template_ids])
        
        if total_length > self.max_seq_length:
            for i, key in enumerate(template_string_keys):
                if key == self.truncation_field:
                    assert len(template_ids[i]) > total_length - self.max_seq_length, "The truncation field is not long enough to truncate"
                    if self.truncation_method == "left":
                        template_ids[i] = template_ids[i][total_length - self.max_seq_length:]
                    else:
                        template_ids[i] = template_ids[i][:-(total_length - self.max_seq_length)]
                    break
        
        text_start_idx = sum(len(ids) for ids in template_ids[:-1]) - 1
        token_ids = [id for ids in template_ids for id in ids] + [self.tokenizer.eos_id]
        return {
            'audio': audio,
            'token_ids': token_ids,
            'token_length': len(token_ids) - 1,
            'answer_start_idx': text_start_idx,
        }
        
        