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
  - The reads are used for producing metagenome assemblies & generating bam files for the contigs
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

Set `real_contigs_simulated_reads: True` for this senario.

* Required columns:
  * `Taxon`
    * Unique names for ground-truth genome assemblies
  * `Fasta`
    * Genome assembly fasta file path
  * `Sample`
    * Unique name for metagenome 
  * `Read1`
    * Path to the forward read fastq file
  * `Read2`
    * Path to the reverse read fastq file

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

### config.yaml

#### Important parameters (`params:`)

##### Simulations

* `MGSIM:`
  * `sim_reps:`
    * Number of simulation replicates
  * `community:`
    * `richness:`
      * Fraction of the reference genomes used for each community simulation
        * e.g., 0.5 = 50 of 100 genomes
      * A different subset of references are used for each simulation
    * `abundance_distribution:`
      * Log-normal abundance distribution parameters
    * `random_seed:`
      * Per-community simulation random seed
    * `other_sim_params:`
      * Parameters provided to `MGSIM communities`
  * `reads:`
    * `length:`
      * Read length (bp); multiple values allowed
    * `depth:`
      * Sequencing depth (no. of reads); multiple values allowed
    * `other_sim_params:`
      * Other parameter provided to `art_illumina`
    * `skewer:`
      * Params provided to [skewer](https://github.com/relipmoc/skewer) for read quality control
    * `keep_reads:`
      * Save the fastq files in the `output_dir`?
  * `nonpareil:`
    * `params:`
      * Parameters provided to [nonpareil](https://github.com/lmrodriguezr/nonpareil)
    * `summary:`
      * Parameters provided to `./bin/scripts/nonpareil_summary.R`
  * `assemblers:`
    * `metaspades:`
      * Parameters provided to [metaspades](https://github.com/ablab/spades)
      * Use "Skip" to skip the assembler
    * `megahit:`
      * Parameters provided to [megahit](https://github.com/voutcn/megahit)
      * Use "Skip" to skip the assembler
  * `contigs:`
    * `length_cutoff:`
      * Filter out all contigs < `length_cutoff`
      * **NOTE: this also applies to real-data contigs!**
  * `asmbl_errors:`
    * `metaquast:`
      * Parameters provided to [metaquast](https://github.com/ablab/quast)
      * Metaquast is used to determine the ground truth for simulated data
        * The contigs are compared to the reference genomes used for the simulations
    * `keep_genomes:`
      * Keep all metaquast info
  * `map:`
    * `samtools:`
      * Parameters provided to [samtools view](https://github.com/samtools/samtools)
    * `bowtie2:`
      * Parameters provided to [bowtie2](https://github.com/BenLangmead/bowtie2)
    * `max_coverage:`
      * The bowtie2 BAM files will be subsampled to this max coverage
        * Mapped reads are subsampled to reduce the max coverage to the specified cutoff
    * `keep_bam:`
      * Keep the BAM files?
      * WARNING: this requires a lot of disk space!
    * `keep_faidx:`
      * Keep the contig reference faidx files?
    * `create_bigwig:`
      * Create bigwig files from the BAM files?
        * Useful for plotting coverage & mapped reads
  * `feature_table:`
    * `make:`
      * Parameters provided to `./bin/scripts/bam2feat`
      * `bam2feat` produces the feature tables from the BAM files
  * `SotA:`
    * `ALE:`
      * Parameters provided to [ALE](https://doi.org/10.1093/bioinformatics/bts723)
      * Use "Skip" to skip 
    * `VALET:`
      * Parameters provided to [VALET](https://github.com/marbl/VALET)
      * Use "Skip" to skip 
    * `metaMIC:`
      * `extract:`
        * Parameters provided to [metaMIC extract](https://github.com/ZhaoXM-Lab/metaMIC)
        * Use "Skip" to fully skip `metaMIC`
      * `predict:`
        * Parameters provided to [metaMIC predict](https://github.com/ZhaoXM-Lab/metaMIC)
  * `real_contigs_simulated_reads:`
    * If providing real contigs, but simulated reads (Scenario 2)

##### Real data (contigs/MAGs)

* Real-data specific parameters (`nonsim_params:`)
  * `subsample_reads:`
    * Subsample reads to this max number of read pairs

* NOTE: some `params:` also do apply to real-data:
  * `contigs:`
  * `map:`
  * `feature_table:`
  * `SotA:`


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
  

