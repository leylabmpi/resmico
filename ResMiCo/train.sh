eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

CODE_PATH="/cluster/home/ddanciu/ResMiCo"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory
MAX_LEN=500

current_time=$(date "+%Y-%m-%d-%H:%M:%S")
log_file="${OUT_PATH}/resmico_${MAX_LEN}_${current_time}.log"
lsf_log_file="${OUT_PATH}/resmico_${MAX_LEN}_${current_time}.lsf.log"

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/ResMiCo"
python setup.py build_ext --inplace

echo "ResMiCo training script. Creating job and submitting to bsub..."
echo "Logging data to ${log_file}"

suffix="_chunked" # set to "_chunked" if training on contig chunks, empty ("") otherwise
if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing MAX_LEN=500 and adding --chunks to parameter list"
  MAX_LEN=500
  additional_params="--chunks"
fi
cmd1="echo Creating scratch directory...; rm -rf /scratch/features_${USER}/; mkdir /scratch/features_${USER}"

cmd2="echo Copying data to local disk...; \
      find ~/deepmased/data/v2/resmico-sm/GTDBr202_n9k_train/features \
        -type f -name stats -o -name toc${suffix} -o -name features_binary${suffix} \
        | xargs -i cp --parents {} /scratch/features_${USER}/"

# defining various sets of features
features_small="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match" 
features20="ref_base num_query_A num_query_C num_query_G num_query_T coverage num_proper_Match num_orphans_Match max_insert_size_Match mean_insert_size_Match min_insert_size_Match stdev_insert_size_Match mean_mapq_Match min_mapq_Match stdev_mapq_Match seq_window_perc_gc num_proper_SNP"
features23="ref_base num_query_A num_query_C num_query_G num_query_T coverage num_proper_Match num_orphans_Match max_insert_size_Match mean_insert_size_Match min_insert_size_Match stdev_insert_size_Match mean_mapq_Match min_mapq_Match stdev_mapq_Match mean_al_score_Match min_al_score_Match stdev_al_score_Match seq_window_perc_gc num_proper_SNP"

cmd3="/usr/bin/time python ResMiCo train --binary-data --feature-files-path /scratch/features_${USER}/ \
      --save-path /cluster/home/ddanciu/tmp --n-procs 8 --log-level info \
      --batch-size 300 --n-fc 1 --num-blocks 4 --fraq-neg 0.1  ${additional_params}  \
      --max-len ${MAX_LEN} --cache --gpu-eval-mem-gb=1 --features ${features_small}"

cd "${CODE_PATH}"

echo "Training command is: ${cmd3}"

# submit the job
bsub -W 2:00 -n 8 -J ResMiCo-n9k -R "span[hosts=1]" -R rusage[mem=50000,ngpus_excl_p=1,scratch=30000] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd1}; ${cmd2}; ${cmd3} 2>&1 | tee ${log_file}"
