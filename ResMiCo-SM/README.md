# ResMiCo-SM

Snakemake pipeline for 2 main purposes:

1. Generating synthetic metagenome data from a set of reference genomes
2. Creating feature tables for ResMiCo training/testing/application

## Setup

You just need a conda env with snakemake & pandas installed.
Snakemake will install all other dependencies via conda.

## Input

### Genomes table

This tab-delimited table differs in which columns are required depending
on the scenario:

1. Simulations from ground-truth reference genomes. 
  - Reads are simulated from the genomes
  - Metagenome communities are simulated (taxon abundances)
2. Simulations using use-generated reads & reference genomes (no read simulation).
  - The reads are assumed to be pre-generated via simulation
  - Ground-truth reference genomes are still provided
  - This is useful for testing other simulation datasets (e.g., CAMI)
3. Generating ResMiCo input feature tables for real genome assemblies (e.g., MAGs).
  - Required in order to run ResMiCo on real data (no ground truth)

#### Scenario 1 (simulating metagenome reads)

* Required columns:
  * `Taxon`
    * Unique name for the genome assembly
  * `Fasta`
    * Genome assembly fasta file path
    
#### Scenario 2 (user-provided reads for the simulation)

* Required columns:
  * `Taxon`
  * `Fasta`
  * `Sample`
  * `Read1`
  * `Read2`

#### Scenario 3 (processing real assemblies)

* Required columns:
  * `Taxon`
    * Unique name for the (meta)genome assembly
  * `Fasta`
    * (Meta)genome assembly fasta file path (contigs)
  * `Sample`
    * Unique name for the metagenome (raw or QC'd reads)
  * `Read1`
    * Path to the forward read fastq file
  * `Read2`
    * Path to the reverse read fastq file



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
  

