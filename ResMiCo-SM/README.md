# ResMiCo-SM

Snakemake pipeline for 2 main purposes:

1. Generating synthetic metagenome data from a set of reference genomes
2. Creating feature tables for ResMiCo training/testing/application

## setup

You just need a conda env with snakemake & pandas installed.
Snakemake will install all other dependencies via conda.

## Input

### Genomes table

* Required
* Tab-delimited table
* Must have the following columns:
  * `Taxon`
    * Taxon name
  * `Fasta`
    * Genome assembly fasta file path

### Reads table

* Only needed if using pre-generated reads (instead of simulating them in the pipeline)
* Tab-delimited table
* Must have the following columns:
  * `Sample`
    * Sample name
  * `Read1`
    * Read1 fastq file path
  * `Read2`
    * Read2 fastq file path

### bam2feat

The executable that converts BAM files to the feature tables used for ResMiCo is written in C.
A precompiled version is located at `./bin/scripts/bam2feat`.
If you need/want to compile a new version, see the README.md in `./feature_extractor/`.

