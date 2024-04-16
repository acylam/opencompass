#!/usr/bin/env sh
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mnt/petrelfs/share_data/zhoufengzhe/model_weights/hf_hub/

cd ~/projects/opencompass
conda activate opencompass

# python run.py ../configs/eval_subj_dynamic_judge_glm4.py \
#     --mode eval \
#     --reuse latest \
#     --work-dir ../outputs/subj_dynamic_judge_glm4/
    # --debug \
    # --dry-run \
    # --debug \
    # --mode infer \
    # --reuse latest

python run.py ../configs/eval_subj_dynamic_judge_gpt4.py \
    --mode eval \
    --reuse latest \
    --work-dir ../outputs/subj_dynamic_judge_gpt4/
#     # --debug \
#     # --dry-run \
#     # --debug \
#     # --mode infer \
#     # --reuse latest
