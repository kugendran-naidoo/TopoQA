# move HAF2 files in pdb directory to parent directory
# DO NOT RUN TWICE

export HAF_LOC="/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets/HAF2"
export SOURCE_DECOY_DIR="decoy"

find ${HAF_LOC}/${SOURCE_DECOY_DIR} -type f |

while read -r decoy_file_name
do

   file_name="${decoy_file_name##*/}"

   # full path excluding file name
   pdb_path=${decoy_file_name%/*}
   target_path=${pdb_path%/*}

   printf "${decoy_file_name}\n"
   printf "${pdb_path}\n"
   printf "${target_path}\n"

   mv ${decoy_file_name} ${target_path} &&
   rmdir ${pdb_path} 2>/dev/null

done 
