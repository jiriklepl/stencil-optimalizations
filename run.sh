echo -e "\033[33mCompiling...\033[0m"
./compile.sh

if [ $? -ne 0 ]; then
    echo -e "\033[31mCompilation failed. Exiting...\033[0m"
    exit 1
fi

echo -e "\n\033[33mRunning...\033[0m"
cd build/src

srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:V100 --time=2:00:00 ./stencils