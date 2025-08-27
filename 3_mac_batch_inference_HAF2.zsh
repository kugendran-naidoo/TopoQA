# 1.) searches a particular dataset directory
# 2.) extracts list of targets (example 3SE8, 3WD5)
# 3.) submits targets to TopoQA inference
# 4.) write inference results to TOPO_RESULTS_DIR/DATASET_NAME/TARGET

export DATASET_NAME="HAF2"
export DATASET_DIR="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets/HAF2"
export DECOY_DIR="decoy"
export TOPO_WORK_DIR="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/TopoQA/topoqa_work"
export TOPO_RESULTS_DIR="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/TopoQA/topoqa_results"
export TOPO_INF_MODEL="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/TopoQA/model/topoqa.ckpt"

# inference configuration
export TOPO_JOBS=8
export TOPO_NUM_WORKERS=2
export TOPO_BATCH_SIZE=4
export TOPO_DEVICE="cpu"
# CUTOFF 10 angstroms for contact metric
export TOPO_CUTOFF=10                  
# accept true or yes - case insensitive
export TOPO_RESULTS_OVERWRITE="yes"
export DSSP=/opt/homebrew/bin/mkdssp

# search topo_results directory for target results
search_decoys() {

   find ${DATASET_DIR}/${DECOY_DIR} -type d |
   sort |    
   grep "/${DECOY_DIR}/"

}

# Main

# timing 
zmodload zsh/datetime 2>/dev/null
typeset -F start_timer elapsed      

typeset -u is_overwrite_set

is_overwrite_set=${TOPO_RESULTS_OVERWRITE}

if [[ ${is_overwrite_set} == TRUE || ${is_overwrite_set} == YES ]]; then
  overwrite=true
else 
  unset overwrite
fi

printf "\nDataset Name: ${DATASET_NAME}\n"

# list all available prediction result files - TOPO_RESULT_FILE
search_decoys |

# process each result per target
while read -r target_decoy_dir
do

    # measure elapsed time
    start_timer=${EPOCHREALTIME}

    # target_name
    target_name="${target_decoy_dir##*/}"


    printf "\nInference Target: ${target_name}\n"

    # run TopoQA inference

    # Macbook Pro M1, 64GB memory
    # jobs 8, num-workers 2, batch-size 4, device cpu (tried, but mps does not work)
    # checkpoint model/topoqa.ckpt
    # cutoff must = 10 - 10^(-9) metres for residue distances in paper

    # print executed statement
    printf '%s\n' \
    "python k_mac_inference_model_refactored.py \\" \
    "    --complex-folder \\" \
    "    ${target_decoy_dir} \\" \
    "    --work-dir \\" \
    "    ${TOPO_WORK_DIR}/${DATASET_NAME}/${target_name} \\" \
    "    --results-dir \\" \
    "    ${TOPO_RESULTS_DIR}/${DATASET_NAME}/${target_name} \\" \
    "    --checkpoint \\" \
    "    model/topoqa.ckpt \\" \
    "    --jobs ${TOPO_JOBS} \\" \
    "    --num-workers ${TOPO_NUM_WORKERS} \\" \
    "    --batch-size ${TOPO_BATCH_SIZE} \\" \
    "    --device ${TOPO_DEVICE} \\" \
    "    --cutoff ${TOPO_CUTOFF} \\" \
    "    ${overwrite:+--overwrite}"

    python k_mac_inference_model_refactored.py \
    --complex-folder \
    ${target_decoy_dir} \
    --work-dir \
    ${TOPO_WORK_DIR}/${DATASET_NAME}/${target_name} \
    --results-dir \
    ${TOPO_RESULTS_DIR}/${DATASET_NAME}/${target_name} \
    --checkpoint \
    model/topoqa.ckpt \
    --jobs ${TOPO_JOBS} \
    --num-workers ${TOPO_NUM_WORKERS} \
    --batch-size ${TOPO_BATCH_SIZE} \
    --device ${TOPO_DEVICE} \
    --cutoff ${TOPO_CUTOFF} \
    ${overwrite:+--overwrite}

    elapsed=$(( EPOCHREALTIME - start_timer ))

    printf "Elapsed: %.0f s\n" "${elapsed}"
 
done
