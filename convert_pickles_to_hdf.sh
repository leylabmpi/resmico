#!/usr/bin/env bash
source activate deepmased

if [ "$#" -ne 3 ]; then
    echo "Required arguments: input_dir, output_dir, number of processes"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BASE_DIR=$1
OUT_DIR_BASE=$2
NPROC=$3

echo "Converting pickles from $1 writing to $2 using $3 processes"

cd ${BASE_DIR}
find . -name '*.pkl' | xargs --max-procs=${NPROC} -I {} python ${SCRIPT_DIR}/convert_pickle.py ${BASE_DIR}/{} ${OUT_DIR_BASE}/$(echo "{}.h5")

# command to run
# bsub -n 4 -R rusage[mem=70000] $(pwd)/convert_pickles_to_hdf.sh /cluster/home/omineeva/global_projects/projects/projects2019-contig_quality/data/v2/GTDBr95_r10_novel-genus/features /cluster/home/omineeva/global_projects/projects/projects2019-contig_quality/data/v2/GTDBr95_r10_novel-genus/features_h5 4