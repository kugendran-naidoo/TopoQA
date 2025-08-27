# Macbook Pro M1, 64GB memory
# jobs 8, num-workers 2, batch-size 4, device cpu (tried, but mps does not work)
# checkpoint model/topoqa.ckpt
# cutoff must = 10 - 10^(-9) metres for residue distances in paper

export DSSP=/opt/homebrew/bin/mkdssp

# SET num-workers=0 for pdb trace
# SET jobs=1 for pdb trace
# SET batch-size=1 for pdb trace

python k_mac_inference_model_refactored.py.debugging \
--complex-folder \
  /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/data/DProQ_benchmark/BM55-AF2/decoy/3SE8 \
--work-dir \
  debug_work/3SE8 \
--results-dir \
  debug_results/3SE8 \
--checkpoint \
  model/topoqa.ckpt \
--jobs 1 \
--num-workers 0 \
--batch-size 1 \
--device cpu \
--cutoff 10 \
--overwrite
