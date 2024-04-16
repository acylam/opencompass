#!/usr/bin/env sh
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/mnt/petrelfs/share_data/zhoufengzhe/model_weights/hf_hub/

cd ~/projects/opencompass
conda activate opencompass

python run.py /mnt/petrelfs/linjunyao/projects/opencompass/tmp_work/config/infer_tasks.py \
    --mode infer \
    --reuse latest \
    --work-dir /mnt/petrelfs/linjunyao/projects/opencompass/tmp_work/outputs/infer_tasks/
    # --debug 
    # --dry-run \
    # --debug \
    # --mode infer \
    # --reuse latest

# python run.py configs/eval_glm4.py \
#     --mode eval \
#     --reuse latest \
#     --work-dir outputs/compass_arena_dynamic_judge/
#     # --debug \
#     # --dry-run \
#     # --debug \
#     # --mode infer \
#     # --reuse latest
