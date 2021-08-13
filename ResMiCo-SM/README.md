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

## Output

* General directory tree naming format:
  * `$OUTDIR/category/richness/abundance_distribution/simulation_replicate/read_length/sequencing_depth/assembler/`
  * This path is referred to as `$FULLPATH` below
* `$OUTDIR/features/`
  * `feature_files.tsv`
    * A table listing all features; can be directly used as input for ResMiCo
  * `$FULLPATH/features.tsv.gz`
    * Feature tables
* `$OUTDIR/true_errors/`
  * metaQUAST data
* `$OUTDIR/genomes/`
* `$OUTDIR/MGSIM/`
  * Metagenome simulation data
  * `comm_abund.txt`
    * Abundances of taxa
  * `comm_wAbund.txt`
    * Abundances of taxa weighted by genome size (used for read simulations)
  * `comm_beta-div.txt`
    * Beta diversity among communities
* `$OUTDIR/assembly/`
  * Info about the metagenome assemblies
* `$OUTDIR/map/`
  * Info on mapping reads to contigs (e.g., BAM files, if kept)
  * [bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) used for mapping
* `$OUTDIR/coverage/`
  * Info on how much of the total diversity was measured, given the sequencing depth
  * [Nonpareil3](https://doi.org/10.1128/mSystems.00039-18) used for estimations
  

