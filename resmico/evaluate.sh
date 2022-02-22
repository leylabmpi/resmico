eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/7.5 nccl/2.3.7-1

MODEL_NO_COUNT="/cluster/home/ddanciu/tmp/no_count.h5"
MODEL_COUNT="/cluster/home/ddanciu/tmp/count.h5"
MODEL_ALL_FEATURES="/cluster/home/ddanciu/tmp/mc_epoch_40_aucPR_0.727_resmico_10000_2022-01-20_09-30-25_model.h5"
MODEL_9_FEATURES="/cluster/home/ddanciu/tmp/mc_epoch_42_aucPR_0.719_resmico_10000_2022-02-01_15-45-07_model.h5"
MODEL_9_FEATURES_15K="/cluster/home/ddanciu/tmp/mc_epoch_52_aucPR_0.725_resmico_15000_2022-02-01_15-44-40_model.h5"
MODEL_NO_INSERT_SIZE="/cluster/home/ddanciu/tmp/no_insert_size.h5"

MODELS=("${MODEL_ALL_FEATURES}" "${MODEL_NO_COUNT}" "${MODEL_COUNT}" "${MODEL_9_FEATURES}" "${MODEL_9_FEATURES_15K}" "${MODEL_NO_INSERT_SIZE}")
MODEL_NAMES=("all" "no_count" "all_count" "9_features" "9_features_15K" "no_insert_size")

features="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match" # seq_window_perc_gc seq_window_entropy"
features_small="mean_al_score_Match mean_mapq_Match num_orphans_Match mean_insert_size_Match min_al_score_Match num_proper_Match min_insert_size_Match num_proper_SNP coverage"
features_no_count="mean_al_score_Match mean_mapq_Match mean_insert_size_Match min_al_score_Match min_insert_size_Match"
features_count="num_orphans_Match num_proper_Match num_proper_SNP coverage"
features_no_insert="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage mean_mapq_Match"

MODEL_FEATURES[0]="${features}"
MODEL_FEATURES[1]="${features_no_count}"
MODEL_FEATURES[2]="${features_count}"
MODEL_FEATURES[3]="${features_small}"
MODEL_FEATURES[4]="${features_small}"
MODEL_FEATURES[5]="${features_no_insert}"

DATA_DIR_N9K="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/features"
DATA_DIR_NOVEL="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_novel-family_test/features"
DATA_DIR_CAMI_GUT="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-gut/"
DATA_DIR_CAMI_SKIN="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-skin/"
DATA_DIR_CAMI_ORAL="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-oral/"
DATA_DIR_UHGG_OLD="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm-old"
DATA_DIR_UHGG_NO_MAX="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm"
DATA_DIR_UHGG_NEW="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm_run4"
DATA_DIR_UHGG_MAX_10="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/rmc-sm_run5"
DATA_DIR_REAL_ANX="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/anx/n30r2/"
DATA_DIR_REAL_LLMGA="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/UHGG/LLMGQC_r100/LLMGA/rmc-sm/"
DATA_DIR_NOVEL_HIGH_DIVERSITY='/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n2k_novel-family_test-intraSpec'
DATA_DIR_ANIMAL_GUT='/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/real_data_eval/animal-gut/n30r1'
DATA_DIR_LARGE_INSERT='/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/large_insert_size'
DATA_DIR_CAMI_GUT_SELF_ASSEMBLY="/cluster/work/grlab/projects/projects2019-contig_quality/data/CAMI/CAMI2_HMP-gut/short_read/ResMiCo-SM_cami-contigs"
declare -a DATASETS=("${DATA_DIR_REAL_ANX}" "${DATA_DIR_UHGG_OLD}" "${DATA_DIR_UHGG_NO_MAX}" "${DATA_DIR_UHGG_NEW}" "${DATA_DIR_UHGG_MAX_10}" "${DATA_DIR_NOVEL}" "${DATA_DIR_CAMI_GUT}" "${DATA_DIR_CAMI_SKIN}" "${DATA_DIR_CAMI_ORAL}" "${DATA_DIR_REAL_LLMGA}" "${DATA_DIR_NOVEL_HIGH_DIVERSITY}" "${DATA_DIR_ANIMAL_GUT}" "${DATA_DIR_LARGE_INSERT}" "${DATA_DIR_CAMI_GUT_SELF_ASSEMBLY}")
declare -a DATASET_NAMES=("anx" "uhgg_old" "uhgg_no_max" "uhgg_new" "uhgg_max10" "novel" "gut" "skin" "oral" "llmga", "high-diversity", "animal-gut", "novel-high-insertsz" "gut-self")

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


for m in {0..5}; do # "${!MODELS[@]}"; do
  if ((m==0)); then
    stats_file="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/stats.json"
  else
    stats_file="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/stats_cov.json"
  fi
  for i in {13..13}; do # "${!DATASETS[@]}"; do
    current_time=$(date "+%Y-%m-%d_%H-%M-%S")
    name=${MODEL_NAMES[${m}]}_${DATASET_NAMES[${i}]}_${current_time}
    log_file="${OUT_PATH}/${name}.log"
    lsf_log_file="${OUT_PATH}/${name}.lsf.log"

    echo "ResMiCo evaluation script. Creating job and submitting to bsub..."
    echo "Logging data to ${log_file}"

    cmd="/usr/bin/time python resmico evaluate --binary-data --feature-files-path "${DATASETS[${i}]}" \
          --save-path /cluster/home/ddanciu/tmp --save-name evaluate${suffix}_${name} --n-procs 8 --log-level info \
          --model ${MODELS[${m}]} --mask-padding \
          --stats-file ${stats_file} \
          --min-avg-coverage 0 --max-len ${MAX_LEN} --gpu-eval-mem-gb=0.1 --features ${MODEL_FEATURES[${m}]} ${additional_params}"

    cmd2="awk -F\",\" '\$4>0.5' /cluster/home/ddanciu/tmp/evaluate${suffix}_${name}.csv | wc -l"
  cd "${CODE_PATH}" || exit

    echo "Evaluation command is: ${cmd}"
    echo "Count command is: ${cmd2}"

    # submit the job
    bsub -W 4:00 -n 4 -J ${name}-10K -R "span[hosts=1]" -R rusage[mem=5000,ngpus_excl_p=2] -G ms_raets \
       -oo "${lsf_log_file}" "${cmd} 2>&1 | tee ${log_file}; ${cmd2} 2>&1 | tee -a ${log_file}"
    sleep 1
  done
done

