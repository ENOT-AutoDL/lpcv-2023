onnx="best_org_F1.onnx"
engine="best_org_F1.plan"

python tensorrt/tensorrt_builder.py \
    --onnx $onnx \
    --engine $engine \
    --fp16
