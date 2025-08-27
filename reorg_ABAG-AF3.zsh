# move ABAG-AF3 files in ABAG-AF3_structures to decoy
# move group decoys per target into the same directory

export ABAG_AF3_LOC="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets/ABAG-AF3"
export SOURCE_DECOY_DIR="ABAG-AF3_structures"

typeset -u u_target_name

# make all directories first
find ${ABAG_AF3_LOC}/${SOURCE_DECOY_DIR} -type f |

while read -r decoy_file_name
do

   file_name="${decoy_file_name##*/}"

   temp=${file_name#*_}

   target_name=${temp%%_*}

   u_target_name=${target_name}
 
   # full path excluding file name
   target_path=${decoy_file_name%/*}

   printf "${target_path}/${u_target_name}\n"
   printf "${decoy_file_name}\n"

   # just move file if dir exists
   ls -1 ${target_path}/${u_target_name} 2>/dev/null &&
   mv ${decoy_file_name} ${target_path}/${u_target_name} &&
   printf "moved ${decoy_file_name} to ${u_target_name}\n"

   # create dir if DNE, then move file
   ls -1 ${target_path}/${u_target_name} 2>/dev/null ||
   { mkdir ${target_path}/${u_target_name} &&
     mv ${decoy_file_name} ${target_path}/${u_target_name} 
     printf "created ${u_target_name} and moved ${decoy_file_name}\n"
   }

done 


