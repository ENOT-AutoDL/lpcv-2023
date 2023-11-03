checkpoint="output/lpcv/best_org_F1.pt"
model_name="pidnet_s"

python tools/eval.py \
--cfg configs/lpcv/pidnet_small_lpcv.yaml \
--checkpoint $checkpoint \
--model-name $model_name
