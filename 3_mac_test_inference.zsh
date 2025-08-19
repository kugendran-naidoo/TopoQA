# Macbook Pro M1, 64GB memory
# jobs 8, num-workers 2, batch-size 4, device cpu (tried, but mps does not work)
# checkpoint model/topoqa.ckpt
# cutoff must = 10 - 10^(-9) metres for residue distances in paper

python k_mac_inference_model_refactored.py \
--complex-folder \
  /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/data/DProQ_benchmark/BM55-AF2/decoy/3SE8 \
--work-dir \
  topoqa_work/3SE8 \
--results-dir \
  topoqa_results/3SE8 \
--checkpoint \
  model/topoqa.ckpt \
--jobs 8 \
--num-workers 2 \
--batch-size 4 \
--device cpu \
--cutoff 10 \
--overwrite
