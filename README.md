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

`pytest -s`


# General usage

## ResMiCo-SM

See the [ResMiCo-SM README](./ResMiCo-SM/README.md)

## ResMiCo (DL)

Main interface: `resmico -h`

Note: Although `ResMiCo` can be run on a CPU, the performance is orders of magnitude
wors, so we only recommend running on CPU for testing. 

### Predicting with existing model

See `resmico predict -h` 

### Training a new model

See `resmico train -h` 

### Evaluating a model

See `resmico evalulate -h`

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

## Training

```
resmico train --log-progress --n-procs 4 --save-path model --save-name genomes-n10 --feature-file-table genomes-n10_features/feature_files.tsv
```

## Predict using new model

```
resmico predict --model-path ./model/ --model-name genomes-n10 --save-path predictions --save-name UHGG-n9 --n-procs 4 UHGG-n9_features/feature_files.tsv
```