python - << 'PY'
import importlib.util
for m in ["torch_scatter","torch_sparse","torch_cluster","torch_spline_conv"]:
    print(m, "→", importlib.util.find_spec(m) is not None)
PY

python -c "import torch_geometric; print('PyG OK')"
