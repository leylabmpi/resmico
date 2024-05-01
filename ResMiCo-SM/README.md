# ResMiCo-SM

Snakemake pipeline for 2 main purposes:

1. Generating synthetic metagenome data from a set of reference genomes
2. Creating feature tables for ResMiCo training/testing/application

## Install

### Conda 

You need a conda environment with snakemake & pandas installed.
Snakemake will install all other dependencies via conda.

### ResMiCo-SM

```
git clone --recurse-submodules git@github.com:leylabmpi/ResMiCo.git
cd ResMiCo/ResMiCo-SM/
```

Note: snakemake should run on a local machine without further configuration, but running snakemake on a HPC/cloud setup with require some more configuration. See the snakemake docs for [cluster execution](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) or [cloud execution](https://snakemake.readthedocs.io/en/stable/executing/cloud.html).

## Input

### Example input data

```
wget wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes_n10.tar.gz
wget http://ftp.tue.mpg.de/ebio/projects/ResMiCo/genomes_n10.md5
md5sum --check genomes_n10.md5
tar -pzxvf genomes_n10.tar.gz && rm -f genomes_n10.tar.gz
```

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
    * `contig_frac`
      * If `contig_frac > 1`, it acts as a length cutoff (`contig_length__bp >= contig_frac`).
      * If `contig_frac between 0 and 1`, it acts as a subsampling mechanism: what fraction of contigs to randomly subsample and used as a reference for read mapping?
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


## Run

### Running locally 

```
snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>
```

### Running on a cluster

You will need to setup a snakemake profile specific to your cluster setup.
See the following for how:

* [Ley Lab snakemake profiles](https://github.com/leylabmpi/snakemake_profiles)
* [Snakemake docs on cluster config](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html)
* [Official snakemake profiles](https://github.com/Snakemake-Profiles)

Note that the job submission script should include the following resources:

* `time` = max job time in minutes
* `n` = number of threads requested
* `mem_gb_pt` = per-thread mem in gigabytes

#### Running on an SGE cluster

The SGE profile in [Ley Lab snakemake profiles](https://github.com/leylabmpi/snakemake_profiles)
should work without any modifications (read the README).

You can use `./snakemake_sge.sh` for convenience

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
  

### Features table

> Note: All columns ending in `_*` appear in the table twice: once as `*_Match` (reads that matched the reference at that position) and once as `*_SNP`, which are reads that didn't match the reference at that position.

* **Basic info**
  * `assembler`
    * metagenome assembler used
  * `contig`
    * contig ID
  * `position`
    * position on the contig (bp)
  * `ref_base`
    * nucleotide at that position on the contig
* **Extracted from the bam file**
  * `num_query_A`
    * number of reads mapping to that position with 'A'
  * `num_query_C`
    * number of reads mapping to that position with 'C'
  * `num_query_G`
    * number of reads mapping to that position with 'G'
  * `num_query_T`
    * number of reads mapping to that position with 'T'
  * `num_SNPs`
    * number of SNPs at that position
  * `coverage`
    * number of reads mapping to that position
  * `num_discordant`
    * number of reads in which:
      * the read belongs to a pair
      * the read mate is not properly mapped (see pysam definition)
  * `min_insert_size_*`
    * minimum paired-end read insert size for all reads mapping to that position
  * `mean_insert_size_*`
    * mean paired-end read insert size for all reads mapping to that position
  * `stdev_insert_size_*`
    * stdev paired-end read insert size for all reads mapping to that position
  * `max_insert_size_*`
    * max paired-end read insert size for all reads mapping to that position
  * `min_mapq_*`
    * minimum read mapping quality for all reads mapping to that position
  * `mean_mapq_*`
    * mean read mapping quality for all reads mapping to that position
  * `stdev_mapq_*`
    * stdev read mapping quality for all reads mapping to that position
  * `max_mapq_*`
    * max read mapping quality for all reads mapping to that position
  * `num_proper_*`
    * number of reads mapping to that position with proper read pairing
  * `num_diff_strand_*`
    * number of reads mapping to that position where mate maps to the other strand
    * "proper" pair alignment determined by bowtie2
  * `num_orphans_*`
    * number of reads mapping to that position where the mate did not map
  * `num_supplementary_*`
    * number of reads mapping to that position where the alignment is supplementary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
  * `num_secondary_*`
    * number of reads mapping to that position where the alignment is secondary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
  * `num_discordant_*`
    * See `num_discordant` above 
  * `seq_window_entropy`
    * sliding window contig sequence Shannon entropy
    * window size defined with the `make_features:` param in the `config.yaml` file
  * `seq_window_perc_gc`
    * sliding window contig sequence GC content
    * window size defined with the `make_features:` param in the `config.yaml` file
* **MetaQUAST info**
  * `Extensive_misassembly`
    * the "extensive misassembly" classification set by MetaQUAST
    * encoding: `1 = misassembly; 0 = no misassembly`
  * `Extensive_misassembly_by_pos`
    * Per-contig-position labeling of misassembly types set by MetaQUAST
    * Note: multiple misassembly labels are possible per position (eg, 'inversion,translocation')


#### Features file table

This is a table automatically generated by `ResMiCo-Sm`, which
lists all individual feature tables and their associated metadata
(e.g., simulation parameters).

Run `resmico train -h` to get a full description.
