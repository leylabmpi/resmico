eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

DATA_DIR_N9K="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/features"
DATA_DIR_NOVEL="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_novel-family_test/features"

CODE_PATH="/cluster/home/ddanciu/resmico"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory
MAX_LEN=5000
MODEL="/cluster/home/ddanciu/tmp/mc_epoch_24_aucPR_0.639_resmico_model.h5"
SCRATCH_DIR="/scratch/features_${USER}/"

suffix="" # "_chunked" # set to "_chunked" if evaluating on contig chunks, empty ("") otherwise
if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing MAX_LEN=500 and adding --chunks to parameter list"
  MAX_LEN=500
  additional_params="--chunks"
fi

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="${OUT_PATH}/evaluate_novel_${MAX_LEN}_${current_time}.log"
lsf_log_file="${OUT_PATH}/evaluate_novel_${MAX_LEN}_${current_time}.lsf.log"

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/resmico"
python setup.py build_ext --inplace


echo "ResMiCo evaluation script. Creating job and submitting to bsub..."
echo "Logging data to ${log_file}"

features_small="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match seq_window_perc_gc" # seq_window_entropy"

features_smaller="num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match seq_window_perc_gc seq_window_entropy"

cmd1="echo Creating scratch directory...; rm -rf ${SCRATCH_DIR}; mkdir ${SCRATCH_DIR}"
cmd2="echo Copying data to local disk...; \
      find -L ${DATA_DIR} \
        -type f -name stats -o -name toc${suffix} -o -name features_binary${suffix} \
        | xargs -i cp --parents {} ${SCRATCH_DIR}"

cmd3="/usr/bin/time python resmico evaluate --binary-data --feature-files-path ${SCRATCH_DIR} \
      --save-path /cluster/home/ddanciu/tmp --save-name evaluate${suffix}_${MAX_LEN} --n-procs 8 --log-level info \
      --model ${MODEL} \
      --max-len ${MAX_LEN} --gpu-eval-mem-gb=2 --features ${features_small} ${additional_params}"
#      --val-ind-f ${DATA_DIR}/val_ind.csv"

cmd4="echo Cleaning scratch directory...; rm -rf ${SCRATCH_DIR}"

cd "${CODE_PATH}"

echo "Evaluation command is: ${cmd3}"

# submit the job
bsub -W 4:00 -n 4 -J eval-novel -R "span[hosts=1]" -R rusage[mem=20000,ngpus_excl_p=2,scratch=80000] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd1}; ${cmd2}; ${cmd3} 2>&1 | tee ${log_file}; ${cmd4}"