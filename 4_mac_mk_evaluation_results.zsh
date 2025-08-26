# 1.) searches topoqa_results directory
# 2.) extracts targets (example 3SE8)
# 3.) constructs <target>.unified_results.csv from each target's results.csv
#     and joining it to the ground truth for that target within label_info.csv
#     in the structure:
#     target, model, pred_dockq, true_dockq

export GROUND_TRUTH_FILE="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/data/DProQ_benchmark/BM55-AF2/label_info.csv"
export TOPO_RESULTS_DIR="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/TopoQA/topoqa_results"
export TOPO_RESULT_FILE="result.csv"

# search topo_results directory for target results
result_file=$(find ${TOPO_RESULTS_DIR} |
              grep ${TOPO_RESULT_FILE}
             )

# result file name
file="${result_file##*/}"

# full path excluding file name
full_path="${result_file%/*}"

# extract target name
target="${full_path##*/}"

printf "$result_file $target $full_path\n"

# generate modified result file to prepare for joining
temp_result=$(sort ${result_file} |
              # remove header
              grep -v ^MODEL |
              sed "s/^/${target}\,/" |
              sed "s/_tidy,/\,/"
             )

# extract target ground truth
temp_g_truth=$(grep ^${target} ${GROUND_TRUTH_FILE} |
               sort
              )

printf "Result File\n"
printf "${temp_result}\n"

echo
echo

printf "Ground Truth File\n"
printf "${temp_g_truth}\n"

echo
echo

printf "Generated ${target}.unified_results.csv\n"

awk -F, 'FNR==NR {
  k = $1 FS $2
  # remember col3 from File 1 for this key (first occurrence wins)
  if (!(k in a)) a[k] = $3
  next
}
{
  k = $1 FS $2
  # print one joined row and ensure only one output per key
  if (k in a) { print $1","$2","a[k]","$3; delete a[k] }
}' <( printf "${temp_result}\n" ) \
   <( printf "${temp_g_truth}\n" ) > ${full_path}/temp_file.txt

{ printf "target, model, pred_dockq, true_dockq\n"; cat ${full_path}/temp_file.txt; } > ${full_path}/${target}.unified_results.csv && rm ${full_path}/temp_file.txt
