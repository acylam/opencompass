#!/usr/bin/env sh

cd ~/projects/opencompass
# conda init bash
conda activate opencompass

python run.py configs/eval_glm4.py \
    --mode eval \
    --reuse latest
    # --debug \
    # --dry-run \
    # --debug \
    # --mode infer \
    # --reuse latest

# python run.py configs/eval_debug_subjective_compassarena_eng.py \
#     --mode eval \
#     --reuse latest \
    # --debug \
    # --dry-run \
    # --debug \
    # --mode infer \
    # --reuse latest

# python \
#     -u run.py \
#     --launcher="slurm" \
#     --datasets ceval_gen \
#     --hf-path internlm/internlm2-chat-7b \
#     --model-kwargs device_map='auto' \
#     --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
#     --max-seq-len 2048 \
#     --max-out-len 16 \
#     --batch-size 4  \
#     --num-gpus 1  # Number of minimum required GPUs

# python run.py \
#     --datasets siqa_gen winograd_ppl \
#     --hf-path facebook/opt-125m \
#     --model-kwargs device_map='auto' \
#     --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
#     --max-seq-len 2048 \
#     --max-out-len 100 \
#     --batch-size 128  \
#     --num-gpus 1  # Number of minimum required GPUs
