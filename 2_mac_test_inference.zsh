# 1) Force interface extraction to run serially
export TOPOQA_N_JOBS=1        # if you wired this env var, otherwise see code path below

# 2) Force DataLoader to be single-worker
export TOPOQA_NUM_WORKERS=0   # again, only if your script reads it; otherwise code change below

# 3) (Optional) Tell joblib to use threads if you keep n_jobs>1 later
export JOBLIB_MULTIPROCESSING=0

python k_mac_inference_model.py -c /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/data/DProQ_benchmark/BM55-AF2/decoy/3SE8 -w topoqa_work/3SE8 -r topoqa_results/3SE8
