# 1.) searches topoqa_results directory
# 2.) extracts targets (example 3SE8)
# 3.) constructs <target>.unified_results.csv from each target's results.csv
#     and joining it to the ground truth for that target within label_info.csv
#     in the structure:
#     target, model, pred_dockq, true_dockq

export DATASET_NAME="ABAG-AF3"
export GROUND_TRUTH_FILE="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets/ABAG-AF3/label.txt"
export TOPO_RESULTS_DIR="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/TopoQA/topoqa_results/ABAG-AF3"
export TOPO_RESULT_FILE="result.csv"

# search topo_results directory for target results
search_topo_results() {

   find ${TOPO_RESULTS_DIR} |
   sort |    
   grep "/${TOPO_RESULT_FILE}"

}


# accepts 2 parameters - file 1 and file 2
# file 1: modify target results - from TopoQA 
 # sample structure file 1: <target>,<model_name>,<DockQ prediction>
 # sample structure file 1: 3SE8,model_1_multimer_20220423_555805,0.27934432
# file 2: ground truth - file from BM55-AF2 dataset
 # sample structure file 2: <target>,<model_name>,<True DockQ>,<CAPRI>
 # sample structure file 2: 3SE8,model_1_multimer_20220423_555805,0.720,2
# performs join between predicted and ground truth files
# more reliable than unix join
join_results_g_truth() {

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
}' <( printf "${1}\n" ) \
   <( printf "${2}\n" )

}


# address issue with ABAG-AF3 ground truth file
# function to add TARGET_NAME to front of every row
transform_ground_truth() {

   typeset -u target_name

   grep -v ^MODEL ${GROUND_TRUTH_FILE} |
   sort |
   while read -r g_truth_line
   do

      model_name=$(printf "${g_truth_line}\n" |
                   cut -d " " -f 1
                  )

      tmp=${model_name#*_}
      target_name=${tmp%%_*}

      printf "${target_name} ${g_truth_line}\n"

   done

}


# Main

# ABAG-AF3 does not have target on each line - add it
# use this variable later on in main loop
trans_ground_truth=$(transform_ground_truth)

# list all available prediction result files - TOPO_RESULT_FILE
search_topo_results |

# process each result per target
while read -r target_result_file
do

   # result file name
   # not used
   file_name="${target_result_file##*/}" 

   # full path excluding file name
   target_path="${target_result_file%/*}"

   # extract target name
   target_name="${target_path##*/}"

   # modify target result file to prepare for joining
   # to ground truth
   # ABAG-AF3 decoy don't have extra "_tidy" to remove like BM55-AF2
   # removed extra sed found in BM55 script version
   temp_result=$(sort ${target_result_file} |
                # remove header
                grep -v ^MODEL |
                sed "s/^/${target_name}\,/"
               )

   # ABAG-AF3 does not have target on each line - add it
   # needed for join operation later on
   # use trans_ground_truth calculated above

   # extract target ground truth
   # ABAG-AF3 is space delimited - convert to comma
   temp_g_truth=$(printf "${trans_ground_truth}\n" |
                  grep ^${target_name} |
                  sort |
                  sed "s/ /\,/g"
                 )

   printf "\nTarget: ${target_name}\n"
   printf "Result File: ${target_result_file}\n"
   
   printf "Result Dir: ${target_path}\n"
   printf "Evaluation Result File: ${target_name}.unified_results.csv\n"
   printf "Evaluation Ranking Loss File (Audit): ${target_name}.ranking_loss_result.csv\n"

   # relate predicted results to ground truth
   # generate evaluation results used for computing ranking loss
   { printf "TARGET, MODEL, PRED_DOCKQ, TRUE_DOCKQ\n"
     join_results_g_truth "${temp_result}" "${temp_g_truth}" 
   } > ${target_path}/${target_name}.unified_result.csv

   # calculate m* (best true DockQ)
   # from ${target_path}/${target_name}.unified_result.csv above
   m_star=$(LC_ALL=C sort -t, -k4,4nr ${target_path}/${target_name}.unified_result.csv | 
            head -1 |
            cut -d "," -f 4
           )

   # calculate m^ (best predicted DockQ)
   # from ${target_path}/${target_name}.unified_result.csv above
   m_hat=$(LC_ALL=C sort -t, -k3,3nr ${target_path}/${target_name}.unified_result.csv | 
           head -1 |
           cut -d "," -f 4
          )

   printf "ranking loss = m* - m^\n"
   printf "${target_name} m* = ${m_star}\n"
   printf "${target_name} m^ = ${m_hat}\n"

   # calculate ranking loss
   ranking_loss=$(awk -v a="$m_star" -v b="$m_hat" 'BEGIN{print a-b}' | sed "s/,/\./")

   printf "${target_name} ranking loss = ${ranking_loss}\n\n"

   # write ranking loss audit file - to debug if required
   # Begin 
   printf "ranking loss = m* - m^\n" \
   > ${target_path}/${target_name}.ranking_loss_result.csv

   printf "${target_name} m* = ${m_star}\n${target_name} m^ = ${m_hat}\n" \
   >> ${target_path}/${target_name}.ranking_loss_result.csv

   printf "${target_name} ranking loss = ${ranking_loss}\n\n" \
   >> ${target_path}/${target_name}.ranking_loss_result.csv

   m_star_row=$(LC_ALL=C sort -t, -k4,4nr ${target_path}/${target_name}.unified_result.csv | 
                head -1
               )
   printf "${m_star_row}\n" >> ${target_path}/${target_name}.ranking_loss_result.csv

   m_hat_row=$(LC_ALL=C sort -t, -k3,3nr ${target_path}/${target_name}.unified_result.csv | 
               head -1
              )
   printf "${m_hat_row}\n" >> ${target_path}/${target_name}.ranking_loss_result.csv
   # End 

done
