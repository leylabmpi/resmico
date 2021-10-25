eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

DATA_DIR="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/features"
CODE_PATH="/cluster/home/ddanciu/resmico"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory
MAX_LEN=5000
SCRATCH_DIR="/scratch/features_${USER}/"

suffix="_chunked" # set to "_chunked" if training on contig chunks, empty ("") otherwise
if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing MAX_LEN=500 and adding --chunks to parameter list"
  MAX_LEN=500
  additional_params="--chunks"
fi

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="${OUT_PATH}/resmico_${MAX_LEN}_${current_time}.log"
lsf_log_file="${OUT_PATH}/resmico_${MAX_LEN}_${current_time}.lsf.log"

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/resmico"
python setup.py build_ext --inplace


echo "ResMiCo training script. Creating job and submitting to bsub..."
echo "Logging data to ${log_file}"

cmd1="echo Creating scratch directory...; rm -rf ${SCRATCH_DIR}; mkdir ${SCRATCH_DIR}"

cmd2="echo Copying data to local disk...; \
      find ${DATA_DIR} \
        -type f -name stats -o -name toc${suffix} -o -name features_binary${suffix} \
        | xargs -i cp --parents {} ${SCRATCH_DIR}"

# defining various sets of features
features_small="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match seq_window_perc_gc"
features20="ref_base num_query_A num_query_C num_query_G num_query_T coverage num_proper_Match num_orphans_Match max_insert_size_Match mean_insert_size_Match min_insert_size_Match stdev_insert_size_Match mean_mapq_Match min_mapq_Match stdev_mapq_Match seq_window_perc_gc num_proper_SNP"
features23="ref_base num_query_A num_query_C num_query_G num_query_T coverage num_proper_Match num_orphans_Match max_insert_size_Match mean_insert_size_Match min_insert_size_Match stdev_insert_size_Match mean_mapq_Match min_mapq_Match stdev_mapq_Match mean_al_score_Match min_al_score_Match stdev_al_score_Match seq_window_perc_gc num_proper_SNP"

cmd3="/usr/bin/time python resmico train --binary-data --feature-files-path ${SCRATCH_DIR} \
      --save-path /cluster/home/ddanciu/tmp --n-procs 8 --log-level info \
      --batch-size 300 --n-fc 1 --num-blocks 4 --fraq-neg 0.2  ${additional_params}  \
      --val-ind-f ${DATA_DIR}/val_ind.csv --log-progress \
      --max-len ${MAX_LEN} --cache --gpu-eval-mem-gb=1 --features ${features_small} --n-epochs 60"

cmd4="echo Cleaning scratch directory...; rm -rf ${SCRATCH_DIR}"

cd "${CODE_PATH}"

echo "Training command is: ${cmd3}"

# submit the job
bsub -W 4:00 -n 8 -J resmico-n9k -R "span[hosts=1]" -R rusage[mem=25000,ngpus_excl_p=1,scratch=30000] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd1}; ${cmd2}; ${cmd3} 2>&1 | tee ${log_file}; ${cmd4}"
