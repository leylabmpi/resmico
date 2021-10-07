eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1

echo "ResMiCo training script. Creating job and submitting to bsub..."

suffix="_chunked" # set to "_chunked" if training on contig chunks, empty ("") otherwise
cmd1="echo Creating scratch directory...; mkdir /scratch/features"

cmd2="echo Copying data to local disk...; find ~/deepmased/data/v2/resmico-sm/GTDBr202_n9k_train/features \
     -type f -name stats -o -name toc${suffix} -o -name features_binary${suffix} \
     | xargs -i cp --parents {} /scratch/features/"

cmd3="/usr/bin/time python ResMiCo train --binary-data --feature-files-path /scratch/features/ \
      --save-path /cluster/home/ddanciu/tmp --n-procs 8 --log-level info \
      --batch-size 300 --n-fc 1 --num-blocks 4 --fraq-neg 0.1  \
      --max-len 5000 --cache --gpu-eval-mem-gb=1"

# submit the job
bsub -W 6:00 -n 8 -J ResMiCo-n9k -R "span[hosts=1]" -R rusage[mem=50000,ngpus_excl_p=2,scratch=30000] -G ms_raets \
     "${cmd1}; ${cmd2}; ${cmd3}"