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

# Citation

If using ResMiCo in your work, please cite:
[ResMiCo: increasing the quality of metagenome-assembled genomes with deep learning](https://www.biorxiv.org/content/10.1101/2022.06.23.497335v1.abstract)

# Installation
## Using pip (easyest, but works only on specific platforms)
Resmico depends on tensorflow, pandas and numpy, each of which coming with its own host of 
dependency nightmares. We have created pip wheels for python 3.8, 3.9 and 3.10 for both Mac OS (x86) and relatively recent Linux versions 
(using glibc >= 2.31). :
```bash
pip install resmico
```
**Note**: resmico depends on tensorflow, so it won't work on Mac machines with Apple silicon.

If you encounter the following error:
```
ERROR: Could not find a version that satisfies the requirement ResMiCo (from versions: none)
ERROR: No matching distribution found for ResMiCo
```
you may need to upgrade your pip version. Try running:
```bash
pip install --upgrade pip
```
If you encounter
```
ERROR: Ignored the following versions that require a different python version: 1.0.12 Requires-Python >=3.8; 1.0.13 Requires-Python >=3.8; 1.1.0 Requires-Python >=3.8; 1.1.1 Requires-Python >=3.8; 1.2.0 Requires-Python >=3.8; 1.2.1 Requires-Python >=3.8; 1.2.2 Requires-Python >=3.8
ERROR: Could not find a version that satisfies the requirement ResMiCo (from versions: none)
ERROR: No matching distribution found for ResMiCo
```
you need to create an environment with a Python version >=3.8 using e.g. conda:
```bash
conda create -n resmico python=3.8
conda activate resmico
```


## From source using pip
If none of the pip wheels is compatible with your system, you can still install resmico relatively easily by installing
the dependencies via conda (or mamba) and using `pip` to install resmico:
```bash
git clone https://github.com/leylabmpi/resmico
conda env create -n resmico -f resmico/environment.yml
conda activate resmico
pip install resmico
```


> WARNING: the resmico bioconda recipe is currently set to an old version of
resmico. That old version does not match the current user interface
(e.g., lacks `resmico bam2feat`). So, we do not recommend using
the bioconda recipe for installing resmico at this time.

## Running the ResMiCo package tests 

Install `pytest` and `pytest-console-scripts`. For example:

```
mamba install pytest pytest-console-scripts
```

Run tests with `pytest`

```
pytest -s --hide-run-results --script-launch-mode=subprocess ./resmico/tests/
```

# General usage

## ResMiCo-SM snakemake pipeline

Use ResMiCo-SM for creating feature files from real data or simulate new data.

See the [ResMiCo-SM README](./ResMiCo-SM/README.md)

> Note `resmico bam2feat` can also be used to create feature tables from
real data: contig fasta files & associated BAM files (mapped reads)

## ResMiCo package

Main interface: `resmico -h`

Note: Although `ResMiCo` can be run on a CPU, it is orders of magnitude
slower than on a GPU, so we only recommend running on CPUs for testing. 

### Creating feature tables

See `resmico bam2feat -h`

### Predicting with existing model

See `resmico evaluate -h` 

### Filtering out misassembled contigs

See `resmico filter -h`

### Training a new model

See `resmico train -h` 


# Example 1: predicting misassemblies with the "default" model

> If you already have metagenome reads mapped to your contigs,
you can process your own data much like in this example.

> The model was trained with data produced via mapping Illumina
paired-end reads with [Bowtie2](https://github.com/BenLangmead/bowtie2). 

## Working directory

```
mkdir example1 && cd example1
```

## Get the example dataset

The dataset consists of a few UHGG genomes (MAGs) and associated BAM files. 
The BAM files were generated by using Bowtie2 to map the associated
metagenome paired-end reads (from which the MAGs were assembled) to the
MAG contigs. 

So, the input consists of fasta files (contigs) and BAM files (mapped reads).

A simple tab-delimited table is used to map the fasta & BAM files.

**Map file format:**

    * A tab-delim table with the columns (any order is allowed): 
      * `Taxon` => name associated with the fasta file of contigs
      * `Fasta` => path to the fasta file of contigs
      * `Sample` => name associated with the BAM file 
      * `BAM` => path to the BAM file of reads mapped to the contigs in `Fasta`

See the `map.tsv` file for an example.

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_bam2feat.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_bam2feat.md5
md5sum --check UHGG-n9_bam2feat.md5
tar -pzxvf UHGG-n9_bam2feat.tar.gz && rm -f UHGG-n9_bam2feat.*
```

## Convert BAM files to feature tables

Create a feature table for each sorted BAM file:

```
resmico bam2feat --outdir features UHGG-n9_bam2feat/map.tsv
```

> Note: the parameters are the same as used for creating
the "default" model from Mineeva et al., 2022, which is 
critical for getting accurate predictions.

## Predict misassemblies

```
resmico evaluate \
  --min-avg-coverage 0.01 \
  --save-path predictions \
  --save-name default-model \
  --feature-files-path features
```

> Note: `--min-avg-coverage` is set to "0.01" here due to the
abnormally low coverage in these small example BAM files.
**DO NOT** use such a low coverage cutoff with real data.

## Filter contigs

Filter out contigs predicted to be misassembled

```
resmico filter \
  --outdir filtered predictions/default-model.csv \
  UHGG-n9_bam2feat/*.fna.gz
```

> Note: change the `--score-cutoff` parameter to alter the number
of contigs filtered.

# Example2: Training & using a new model

## Working directory

```
mkdir example2 && cd example2
```

## Get the example dataset

Training data: simulated from 10 genomes in the GTDB

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes-n10_features.md5
md5sum --check genomes-n10_features.md5
tar -pzxvf genomes-n10_features.tar.gz && rm -f genomes-n10_features.*
```

Test data: simulated from 9 genomes in the UHGG

```
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/UHGG-n9_features.md5
md5sum --check UHGG-n9_features.md5
tar -pzxvf UHGG-n9_features.tar.gz && rm -f UHGG-n9_features.*
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

## Predict using the "default" model

Using the "default" resmico model from the Mineeva et al., 2022 manuscript.
Prediction on the example test data.
This provides a comparison to our newly trained model.

```
resmico evaluate --n-procs 4 \
  --save-path predictions \
  --save-name default-model \
  --feature-files-path UHGG-n9_features/
```

# Tutorials

See the [wiki](https://github.com/leylabmpi/ResMiCo/wiki)

# Notes

## Benchmarking 

### Model evaluation

Benchmarking `resmico evaluate` on the `CAMI2-gut` dataset:

* One GPU (NVIDIA RTX A5000): 108 +/- 0.7 contigs per second
* One CPU (AMD Epyc): 38.7 +/- 10.3 contigs per second

CAMI2-gut metagenome assembly stats:

* No. of metagemes: 10 
* No. of contigs per sample (1000's): 18 +/- 6.4
* Avg. contig length (kbp): 4.1 +/- 0.9

### Training 

> We highly recommend using multiple GPUs for model training on large datasets, as done in the ResMiCo paper.
  Training on CPUs with such large datasets is not feasbile.
