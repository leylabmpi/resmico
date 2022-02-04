#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tfgpu
module load cuda/11.1.1 cudnn/8.0.5 nccl/2.3.7-1

N9K="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train/"
N9K_1REP="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train_1rep"
N9K_SMALL="/cluster/work/grlab/projects/projects2019-contig_quality/data/v2/resmico-sm/GTDBr202_n9k_train_1rep/GTDBr202_n9k_train/cami_err/features/0.005555/"

declare -a DATA_DIRS=("${N9K}") 
CODE_PATH="/cluster/home/ddanciu/resmico"  # replace with whatever directory your source code is in
OUT_PATH="/cluster/home/ddanciu/tmp" # replace this with the desired output directory

SCRATCH_DIR="/scratch/features_${USER}/"

# command line arguments
suffix="" # "_chunked" # set to "_chunked" if training on contig chunks, empty ("") otherwise
max_len=10000
num_translations=1
max_translation_bases=0

# guess the larges batch size such that the batch fits in GPU memory
if ((max_len <= 10000)); then
  batch_size=300
elif ((max_len <= 15000)); then
  batch_size=200
else
  batch_size=100
fi

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -c|--chunks)
      suffix="_chunked"
      shift # past argument
      ;;
    -m|--max-len)
      max_len="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--num-translations)
      num_translations="$2"
      shift # past argument
      shift # past value
      ;;
    -x|--max-translation-bases)
      max_translation_bases="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

if [ "${suffix}" == "_chunked" ]; then
  echo "Using chunked data, forcing max_len=500 and adding --chunks to parameter list"
  max_len=500
  additional_params="--chunks --cache"
fi

echo "Compiling Cython bindings..."
cd "${CODE_PATH}/resmico" || exit
python setup.py build_ext --inplace

for DATA_DIR in "${DATA_DIRS[@]}" 
do
  IFS=',' read -ra DDRS <<< "${DATA_DIR}"
  for DD in "${DDRS[@]}"; do
    if [ ! -d "${DD}" ]; then
      echo "Directory ${DD} does not exist. Bailing out."
      exit 1
    fi
  done
  current_time=$(date "+%Y-%m-%d_%H-%M-%S")
  log_file="${OUT_PATH}/resmico_${max_len}_${current_time}.log"
  lsf_log_file="${OUT_PATH}/resmico_${max_len}_${current_time}.lsf.log"

  echo "ResMiCo training script. Creating job and submitting to bsub..."
  echo "Data dir is ${DATA_DIR}"
  echo "Logging data to ${log_file}"
  cmd1="echo Creating scratch directory...; rm -rf ${SCRATCH_DIR}; mkdir ${SCRATCH_DIR}"

  cmd2="echo Copying data to local disk...; \
        find -L ${DATA_DIR} \
         -type f -name stats -o -name toc${suffix} -o -name features_binary${suffix} \
         | xargs -i cp --parents {} ${SCRATCH_DIR}"
  # defining various sets of features
  features_small="ref_base num_query_A num_query_C num_query_G num_query_T num_SNPs num_proper_Match num_orphans_Match mean_al_score_Match coverage stdev_insert_size_Match mean_mapq_Match" # seq_window_perc_gc seq_window_entropy

  features_smaller="mean_al_score_Match mean_mapq_Match num_orphans_Match mean_insert_size_Match min_al_score_Match num_proper_Match min_insert_size_Match num_proper_SNP coverage"
  features_no_count="mean_al_score_Match mean_mapq_Match mean_insert_size_Match min_al_score_Match min_insert_size_Match"
  features_count="num_orphans_Match num_proper_Match num_proper_SNP coverage"

  cmd3="/usr/bin/time python resmico train --binary-data --feature-files-path ${SCRATCH_DIR} \
      --save-path /cluster/home/ddanciu/tmp --save-name  resmico_${max_len}_${current_time} \
      --n-procs 8 --log-level info \
      --batch-size ${batch_size} --n-fc 1 --num-blocks 4 --fraq-neg 0.2  ${additional_params}  \
      --max-len ${max_len} --gpu-eval-mem-gb 1 --features ${features_count} --n-epochs 100 \
      --num-translations ${num_translations} --max-translation-bases ${max_translation_bases} \
      --min-avg-coverage 0 --mask-padding --net-type cnn_resnet_avg \
      --lr-init 0.0001 \
      --val-ind-f ${DATA_DIR}/evaluation_indices_fixed.csv \
      --stats-file ${DATA_DIR}/stats_cov.json" # also contains coverage stats
#      --feature-file-match '/1/'"


  cmd4="echo Cleaning scratch directory...; rm -rf ${SCRATCH_DIR}"

  cd "${CODE_PATH}" || exit

  echo "Training command is: ${cmd3}"
  # the large training dataset takes 413GB
  bsub -W 120:00 -n 8 -J 15k-count -R "span[hosts=1]" -R rusage[mem=50000,ngpus_excl_p=4,scratch=80000] -G ms_raets \
     -oo "${lsf_log_file}" "${cmd1}; ${cmd2}; ${cmd3} 2>&1 | tee ${log_file}; ${cmd4}"

  sleep 1
done

