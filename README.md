[![ResMiCo](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml)

![overview](https://user-images.githubusercontent.com/2468572/175315122-1ec3e6e3-419f-4154-af31-21b9dcb2e38f.png)

# Introduction

ResMiCo is a deep learning model capable of detecting metagenome assembly errors. 
ResMiCo's input is summary data derived from re-aligning reads against the putative 
genomes. ResMiCo's output is a number betwen 0 and 1 representing the likelihood that a 
particular genome was misassembled.

The tool is divided into two main parts:


* **ResMiCo-SM**
  * A snakemake pipeline for:
    * creating feature tables from real-world assemblies
	  * input: >=1 fasta of contigs, along with associated Illumina paired-end reads
    * generating train/test datasets from reference genomes
  * See the [ResMiCo-SM README](./ResMiCo-SM/README.md)
* **ResMiCo (DL)**
  * A python package for misassembly detection via deep learning

# Installation

It is possible to install this project using `pip`:
```bash
pip install resmico
```

or `conda`, using the ``bioconda`` channel:
```bash
conda install -c bioconda resmico
```

# Citation

If using ResMiCo in your work, please cite:
> TODO
  

# General usage

## ResMiCo-SM: create feature files

Use ResMiCo-SM for creating feature files from real data or simulate new data.

See the [ResMiCo-SM README](./ResMiCo-SM/README.md)

## ResMiCo (DL): predict misassemblies

Main interface: `resmico -h`

Note: Although `ResMiCo` can be run on a CPU, it is orders of magnitude
slower than on a GPU, so we only recommend running on CPU for testing. 

### Predicting with existing model

See `resmico evaluate -h` 

### Filtering out contigs predicted to be misassembled

See `resmico filter -h`

### Training a new model

See `resmico train -h` 


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

## Running tests

Install `pytest` and `pytest-console-scripts`. For example:

```
mamba install pytest pytest-console-scripts
```

Run tests

```
pytest -s --hide-run-results --script-launch-mode=subprocess ./resmico/tests/
```

## Working directory

```
mkdir tutorial && cd tutorial
```

## Get the example dataset

Training data

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.md5
md5sum --check genomes-n10_features.md5
tar -pzxvf genomes-n10_features.tar.gz && rm -f genomes-n10_features.*
```

Test data

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.md5
md5sum --check UHGG-n9_features.md5
tar -pzxvf UHGG-n9_features.tar.gz && rm -f UHGG-n9_features.*
```

## Predict using a pre-trained model

Using the "default" resmico model from the Mineeva et al., 2022 manuscript.
Prediction on the example test data.

```
resmico evaluate --n-procs 4 \
  --save-path predictions \
  --save-name default-model \
  --feature-files-path UHGG-n9_features/
```

## Filter out contigs predicted to be misassembled

Filter out contigs with prediction scores below a specific cutoff.

```
resmico filter \
  --outdir filtered-contigs \
  predictions/default-model.csv \
  UHGG-n9_features/fasta/*fna.gz
```

> You may need to adjust the `--score-cutoff` in order to filter some contigs

## Training on the example train data

Train a new model with the example train dataset.

```
resmico train --log-progress --n-procs 4 --n-epochs 2 \
  --save-path model-n10 --stats-file='' \
  --save-name genomes-n10 \
  --feature-files-path genomes-n10_features
```

# Tutorials

See the [wiki](https://github.com/leylabmpi/ResMiCo/wiki)
