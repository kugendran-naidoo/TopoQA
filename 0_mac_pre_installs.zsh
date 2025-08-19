# python 3.9.19

python -m pip install --upgrade pip wheel setuptools

# must install this first - fails if done bulk
# --- PyTorch (macOS/MPS wheels; no CUDA, no torchtriton) ---
pip install torch==2.1.0

# ARM issue - have ignore rest of torch requirements
pip install torch_geometric==2.5.3

# Needed
pip install pytorch-lightning

# Fix Numpy mess
pip install --upgrade 'numpy<2'  # e.g., 1.26.4

# test torch, geometric works
python -c "import numpy, torch, torch_geometric; print(numpy.__version__, torch.__version__, 'OK')"

# sanity checks - scatter, sparse, cluster, spline - False on Mac
# geometric OK
zsh ./k_torch_test_mac.py 

# install rest of python libraries
pip install -r requirements_macos_mps.txt

# install mkdssp
brew tap brewsci/bio || true
brew install dssp

# test mkdssp
which mkdssp && mkdssp --version
