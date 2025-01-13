# Fine-tuning Whisper with NVIDIA NeMo and Megatron-Core
This project demonstrates how to fine-tune OpenAI's Whisper model using NVIDIA NeMo and Megatron-Core. With the integration of Tensor Parallelism, this setup enables efficient training for large-scale models, distributing the workload across multiple GPUs, and significantly reducing memory bottlenecks.

## Requirements
- NVIDIA GPUs with support for Tensor Parallelism
- NVIDIA NeMo NGC Docker image: nvcr.io/nvidia/nemo:24.12.
You can pull this image from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/.

## Getting Started
The following steps guide you through setting up the training configuration. For a more hands-on example, you can also refer to the `tutorial.ipynb` notebook, which provides an interactive walkthrough for setting up and running experiments.
1. **Pull docker images:** pull nvcr.io/nvidia/nemo:24.12 from NGC.
```bash
docker pull nvcr.io/nvidia/nemo:24.12
```
2. **Clone the Repository:**
```bash
git clone https://github.com/YK-Fu/Megatron-Whisper.git
cd Megatron-Whisper
```
3. **Download Pre-trained Checkpoint (Optional):** Download a pre-trained Whisper checkpoint from HuggingFace and convert it into NeMo format.
```bash
# If you wish to train from scratch, skip this step.
huggingface-cli download <HF_MODEL_ID> --local-dir <HF_MODEL_DIR>
python checkpoint_converter/whisper_hf_to_nemo.py --hf_input_path <HF_MODEL_DIR> --output_dir <NEMO_MODEL_DIR>
```

5. **Prepare Training Data:** Create a jsonl file with the following format for your training data:
```json
{
    "audio_path": "Path to the audio file", 
    "text": "Corresponding transcription", 
    "task": "<|transcribe|> for transcription, <|translate|> for translation",
    "lang": "Language code for the audio (e.g., <|en|> for English)",
    "prompt": "Previous transcription or prompt text for context (optional)",
    "timestamp": "If timestamps are not provided, set this to <|notimestamps|>"
}
```
7. **Model fine-tuning:** Set the desired configuration and run the fine-tuning script:
```bash
CONFIG_NAME=config
NEMO_CKPT=<NEMO_MODEL_DIR>/model.nemo
TOKENIZER=<HF_MODEL_DIR>

NUM_GPUS=2
NAME=megatron_whisper_ft    # Directory to store training results
TENSOR_PARALLEL=2             # Tensor parallel size

PRECISION=bf16     # Support for 16, bf16, 32
MAX_STEP=50
VAL_INTERVAL=25    # Validate the model every VAL_INTERVAL steps
VAL_NUMS=5         # Number of batches to run for validation step
AMP_O2=True        # For precision in bf16, turn on megatron_amp_O2 for better efficiency

LR=1.0e-4
MICRO_BATCH=4      # Batch size feed to the model in each forward path, reduce it if you face OOM issues
GLOBAL_BATCH=8     # Real logical batch size for updating
FREEZE_ENCODER=True    # Whether to freeze the encoder. It is observed that freezing the encoder sometimes give better results than fully fine-tuning

TRAIN_FILES="[<TRAIN_JSONL>]"    # Training dataset list
TRAIN_PROBS="[1.0]"                             # Training dataset sampling probablilty, it should have the same length with TRAIN_FILES.
VALID_FILES="[<VALID_JSONL>]"
TEST_FILES="[<TEST_JSONL>]"
CACHE_DIR="./cache"    # Directory to store dataset cache. If you modify the jsonl files, you should delete this folder before training

python megatron_whisper_training.py \
    --config-path $CONFIG_DIR \
    --config-name $CONFIG_NAME \
    name=$NAME \
    restore_from_path=$NEMO_CKPT \
    trainer.devices=$NUM_GPUS \
    trainer.precision=$PRECISION \
    trainer.max_steps=$MAX_STEP \
    trainer.val_check_interval=$VAL_INTERVAL \
    trainer.limit_val_batches=$VAL_NUMS \
    model.megatron_amp_O2=$AMP_O2 \
    model.micro_batch_size=$MICRO_BATCH \
    model.global_batch_size=$GLOBAL_BATCH \
    model.tensor_model_parallel_size=$TENSOR_PARALLEL \
    model.feature_extractor.path=$TOKENIZER \
    model.freeze_encoder=$FREEZE_ENCODER \
    model.tokenizer.type=$TOKENIZER \
    model.freeze_encoder=$FREEZE_ENCODER \
    model.data.train_ds.file_names=$TRAIN_FILES \
    model.data.train_ds.index_mapping_dir=$CACHE_DIR \
    model.data.train_ds.concat_sampling_probabilities=$TRAIN_PROBS \
    model.data.validation_ds.index_mapping_dir=$CACHE_DIR \
    model.data.validation_ds.file_names=$VALID_FILES \
    model.data.test_ds.file_names=$TEST_FILES \
    model.data.test_ds.index_mapping_dir=$CACHE_DIR \
    model.optim.lr=$LR
```
8. **Convert Fine-tuned Model Back to HuggingFace (Optional):** After fine-tuning, you can convert the model back to HuggingFace format for distribution:
```
python checkpoint_converter/whisper_nemo_to_hf.py \
    --hf_input_path <HF_MODEL_DIR> \
    --nemo_input_path $NAME/checkpoints/megatron_whisper_ft.nemo \
    --output_path <FT_HF_MODEL_DIR>
```
# Key Features
- Tensor Parallelism: Fine-tuning large models like Whisper is made possible by Megatron-Core’s Tensor Parallelism, which distributes the model across multiple GPUs. This enables training of models that would otherwise not fit into a single GPU’s memory.
- Efficient Training with Megatron-Core: Megatron-Core optimizes training by employing Data Parallelism, Model Parallelism, and Mixed Precision techniques. The backend optimizations help accelerate training, making it more memory efficient and scalable.
## Limitations
- Currently, only Data Distributed Parallelism (DDP) and Tensor Parallelism (TP) are supported. Pipeline Parallelism is not yet available.
- Transformer Engine is also not currently supported. Future updates will address these limitations.