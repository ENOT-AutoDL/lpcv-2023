onnx="path/to/model.onnx"
engine="path/to/engine.plan"

python tools/tensorrt_builder.py \
    --onnx $onnx \
    --engine $engine \
    --fp16
