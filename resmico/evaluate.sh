eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

DATA_DIR_N9K="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/features"
DATA_DIR_NOVEL="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_novel-family_test/features"
DATA_DIR_CAMI_GUT="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-gut/"
DATA_DIR_CAMI_SKIN="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-skin/"
DATA_DIR_CAMI_ORAL="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-oral/"

declare -a DATA_DIRS=("${DATA_DIR_NOVEL}" "${DATA_DIR_CAMI_GUT}")
declare -a DATA_NAMES=("novel" "gut")

CODE_PATH="/cluster/home/ddanciu/resmico"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory
MODEL="/cluster/home/ddanciu/tmp/mc_epoch_30_aucPR_0.690_resmico_model.h5"
MAX_LEN=5000

suffix="" # "_chunked" # set to "_chunked" if evaluating on contig chunks, empty ("") otherwise
if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing MAX_LEN=500 and adding --chunks to parameter list"
  MAX_LEN=500
  additional_params="--chunks"
fi

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/resmico" || exit
python setup.py build_ext --inplace

for i in "${!DATA_DIRS[@]}"; do
  DATA_DIR="${DATA_DIRS[${i}]}"
  name=${DATA_NAMES[${i}]}
  current_time=$(date "+%Y-%m-%d_%H-%M-%S")
  log_file="${OUT_PATH}/${name}_${current_time}.log"
  lsf_log_file="${OUT_PATH}/${name}_${current_time}.lsf.log"

  echo "ResMiCo evaluation script. Creating job and submitting to bsub..."
  echo "Logging data to ${log_file}"

  features_small="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match" # seq_window_perc_gc seq_window_entropy"

  features_smaller="num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match seq_window_perc_gc seq_window_entropy"

  cmd="/usr/bin/time python resmico evaluate --binary-data --feature-files-path ${DATA_DIR} \
        --save-path /cluster/home/ddanciu/tmp --save-name evaluate${suffix}_${MAX_LEN} --n-procs 8 --log-level info \
        --model ${MODEL} --mask-padding \
        --min-avg-coverage 0 --max-len ${MAX_LEN} --gpu-eval-mem-gb=0.1 --features ${features_small} ${additional_params}"

  cd "${CODE_PATH}" || exit

  echo "Evaluation command is: ${cmd}"

  # submit the job
  bsub -W 1:00 -n 4 -J cami-gut -R "span[hosts=1]" -R rusage[mem=20000,ngpus_excl_p=2] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd} 2>&1 | tee ${log_file}"
done