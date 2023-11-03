checkpoint="output/lpcv/best_org_F1.pt"

python tools/export_onnx.py \
    --checkpoint $checkpoint \
    --input-shape 1 3 512 512 \
    --opset 12 --onnx-path best_org_F1.onnx \
