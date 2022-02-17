eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

MODEL_NO_COUNT="/cluster/home/ddanciu/tmp/no_count.h5"
MODEL_COUNT="/cluster/home/ddanciu/tmp/count.h5"
MODEL_ALL_FEATURES="/cluster/home/ddanciu/tmp/mc_epoch_40_aucPR_0.727_resmico_10000_2022-01-20_09-30-25_model.h5"
MODEL_9_FEATURES="~/tmp/9_features.h5"

MODELS=("${MODEL_ALL_FEATURES}" "${MODEL_NO_COUNT}" "${MODEL_COUNT}" "${MODEL_9_FEATURES}")
MODEL_NAMES=("all" "no_count" "all_count" "9_features")

DATA_DIR_N9K="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/features"
DATA_DIR_NOVEL="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_novel-family_test/features"
DATA_DIR_CAMI_GUT="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-gut/"
DATA_DIR_CAMI_SKIN="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-skin/"
DATA_DIR_CAMI_ORAL="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-oral/"
DATA_DIR_REAL_MAG1="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm-old"
DATA_DIR_REAL_MAG2="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm"
DATA_DIR_REAL_MAG3="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm_run4"
DATA_DIR_REAL_MAG4="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm_run5"
DATA_DIR_REAL_ANX="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/anx/"

declare -a DATA_DIRS=("${DATA_DIR_REAL_ANX}") # "${DATA_DIR_REAL_MAG1}" "${DATA_DIR_REAL_MAG2}" "${DATA_DIR_REAL_MAG3}" "${DATA_DIR_REAL_MAG4}" "${DATA_DIR_CAMI_GUT}") #"${DATA_DIR_NOVEL}" "${DATA_DIR_CAMI_GUT}" "${DATA_DIR_CAMI_SKIN}" "${DATA_DIR_CAMI_ORAL}")
declare -a DATA_NAMES=("anx" "uhgg_old" "uhgg_new" "uhgg_run4" "uhgg_run5" "gut") #"novel" "gut" "skin" "oral")

CODE_PATH="/cluster/home/ddanciu/resmico"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory
MAX_LEN=10000

suffix="" # "_chunked" # set to "_chunked" if evaluating on contig chunks, empty ("") otherwise
if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing MAX_LEN=500 and adding --chunks to parameter list"
  MAX_LEN=500
  additional_params="--chunks"
fi

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/resmico" || exit
python setup.py build_ext --inplace


for m in "${!MODELS[@]}"; do
 for i in "${!DATA_DIRS[@]}"; do
  DATA_DIR="${DATA_DIRS[${i}]}"
  name=${DATA_NAMES[${i}]}
  current_time=$(date "+%Y-%m-%d_%H-%M-%S")
  log_file="${OUT_PATH}/${name}_${current_time}.log"
  lsf_log_file="${OUT_PATH}/${name}_${current_time}.lsf.log"

  echo "ResMiCo evaluation script. Creating job and submitting to bsub..."
  echo "Logging data to ${log_file}"

  features="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match" # seq_window_perc_gc seq_window_entropy"

  features_small="mean_al_score_Match mean_mapq_Match num_orphans_Match mean_insert_size_Match min_al_score_Match num_proper_Match min_insert_size_Match num_proper_SNP coverage"
  features_no_count="mean_al_score_Match mean_mapq_Match mean_insert_size_Match min_al_score_Match min_insert_size_Match"
  features_count="num_orphans_Match num_proper_Match num_proper_SNP coverage"
  name=${suffix}_${name}_${current_time}
  cmd="/usr/bin/time python resmico evaluate --binary-data --feature-files-path ${DATA_DIR} \
        --save-path /cluster/home/ddanciu/tmp --save-name evaluate${suffix}_${name}_${current_time} --n-procs 8 --log-level info \
        --model ${MODEL} --mask-padding \
        --stats-file /cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/stats.json \
        --min-avg-coverage 0 --max-len ${MAX_LEN} --gpu-eval-mem-gb=0.1 --features ${features} ${additional_params}"
  
  cmd2="awk -F\",\" '\$4>0.5' /cluster/home/ddanciu/tmp/evaluate${suffix}_${name}_${current_time}.csv | wc -l"  
cd "${CODE_PATH}" || exit

  echo "Evaluation command is: ${cmd}"
  echo "Count command is: ${cmd2}"

  # submit the job
  bsub -W 4:00 -n 4 -J ${name}-10K -R "span[hosts=1]" -R rusage[mem=5000,ngpus_excl_p=2] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd} 2>&1 | tee ${log_file}; ${cmd2} 2>&1 | tee -a ${log_file}"
  sleep 1
 done
done
