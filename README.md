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


# Usage

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

### Filtering out misassembled contigs

See `resmico filter -h`
