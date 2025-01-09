
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $WORK_DIR

EDITED_FILES_DIR="../src/algorithms/an5d/generated-edited"
 
FILES=("gol_32_an5d_host.cu" "gol_32_an5d_kernel.cu" "gol_32_an5d_kernel.hu" "gol_64_an5d_host.cu" "gol_64_an5d_kernel.cu" "gol_64_an5d_kernel.hu")

for FILE in "${FILES[@]}"
do
    echo -e "\e[35mComparing \e[33m$FILE\e[0m"
    diff --color=auto -u $EDITED_FILES_DIR/$FILE $WORK_DIR/"generated"/$FILE
    echo
done
