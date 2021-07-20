# ResMiCo-SM

Snakemake pipeline for 2 main purposes:

1. Generating synthetic metagenome data from a set of reference genomes
2. Creating feature tables for ResMiCo training/testing/application

## setup

You just need a conda env with snakemake & pandas installed.
Snakemake will install all other dependencies via conda.

### bam2feat

The executable that converts BAM files to the feature tables used for ResMiCo is written in C.
A precompiled version is located at `./bin/scripts/bam2feat`.
If you need/want to compile a new version, see the README.md in `./feature_extractor/`.
