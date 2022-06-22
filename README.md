[![ResMiCo](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml)

## Introduction

ResMiCo is a deep learning model capable of detecting metagenome assembly errors. 
ResMiCo's input is summary data derived from re-aligning reads against the putative 
genomes. ResMiCo's output is a number betwen 0 and 1 representing the likelihood that a 
particular genome was misassembled.

## Installation

It is possible to install this project using `pip`:
```bash
pip install resmico
```

or `conda`, using the ``bioconda`` channel:
```bash
conda install -c bioconda resmico
```


## Citation

If using ResMiCo in your work, please cite:
> TODO

## Detailed description

The tool is divided into two main parts:

* **ResMiCo-SM**
  * A snakemake pipeline for:
    * generating ResMiCo train/test datasets from reference genomes
    * creating feature tables from real-world assemblies (contigs and/or MAGs, along with associated Illumina paired-end reads)
* **ResMiCo (DL)**
  * A python package for misassembly detection via deep learning

  
### Running tests

Install `pytest` and `pytest-console-scripts`. For example:

```
mamba install pytest pytest-console-scripts
```

Run tests

```
pytest -s --hide-run-results --script-launch-mode=subprocess ./resmico/tests/
```

# General usage

## ResMiCo-SM

See the [ResMiCo-SM README](./ResMiCo-SM/README.md)

## ResMiCo (DL)

Main interface: `resmico -h`

Note: Although `ResMiCo` can be run on a CPU, the performance is orders of magnitude
wors, so we only recommend running on CPU for testing. 

### Predicting with existing model

See `resmico evaluate -h` 

### Training a new model

See `resmico train -h` 

### Filtering out contigs predicted to be misassembled

See `resmico filter -h`


# Example

## Install resmico 

**Conda/mamba installation:**

```
mamba create -n resmico_env bioconda::resmico
mamba activate resmico_env
```

**...or with pip:**

```
pip install resmico
```

## Working directory

```
mkdir -p tutorial && cd tutorial
```

## Get example dataset

Training data

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.md5
md5sum --check genomes-n10_features.md5
tar -pzxvf genomes-n10_features.tar.gz
```

Test data

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.md5
md5sum --check UHGG-n9_features.md5
tar -pzxvf UHGG-n9_features.tar.gz
```

## Training on example train data

```
resmico train --log-progress --n-procs 4 --n-epochs 2 \
  --save-path model \
  --save-name genomes-n10 \
  --feature-file-table genomes-n10_features/feature_files.tsv
```

## Predict using the newly created model

Prediction on the example test data.

```
MODEL_PATH=/path/to/the/trained/model/it/will/differ/every/time/model.h5
resmico evaluate --binary-data --n-procs 4 \
  --model $MODEL_PATH \
  --save-path predictions \
  --save-name UHGG-n9  \
  --feature-files-path UHGG-n9_features/ \
  --feature-file-table UHGG-n9_features/feature_files.tsv
```

## Filter out contigs predicted to be misassembled

```
resmico filter --score-cutoff 0.03 \
  --outdir filtered-contigs \
  predictions/UHGG-n9.csv \
  UHGG-n9_features/fasta/*fna.gz
```

> You may need to adjust the `--score-cutoff` in order to filter some contigs

## Predict using an pre-trained model

Using the "standard" resmico model from the Mineeva et al., 2022 manuscript.
Prediction on the example test data.

```
MODEL_PATH=/path/to/model/if/not/default/see/github/resmico.h5
resmico evaluate --binary-data --n-procs 4 \
  --model $MODEL_PATH \
  --save-path predictions \
  --save-name UHGG-n9 \
  --feature-files-path UHGG-n9_features/ \
  --feature-file-table UHGG-n9_features/feature_files.tsv
```
