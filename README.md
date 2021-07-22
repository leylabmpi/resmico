[![ResMiCo](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/leylabmpi/ResMiCo/actions/workflows/pythonpackage.yml)

ResMiCo
=========

Residual neural network model to detect metagenome assembly errors



# Citation

> TODO

# Main Description

The tool is divided into two main parts:

* **ResMiCo-SM**
  * A snakemake pipeline for:
    * generating ResMiCo train/test datasets from reference genomes
    * creating feature tables from "real" assemblies (fasta + bam files)
* **ResMiCo (DL)**
  * A python package for misassembly detection via deep learning

# Warnings

* The UI has changed substantially between version 0.2.1 and 0.3.0
  * ResMiCo-Sm generates a feature file table
  * ResMiCo uses the feature file table as input 

# Setup

## conda env

(If needed) Install miniconda (or anaconda)

```
conda env create -f conda_env.yaml -n ResMiCo
conda activate ResMiCo
```

## cloning

```
git clone --recurse-submodules https://github.com/leylabmpi/ResMicCo.git
```

Note the use of submodules. If needed, you can update the submodule(s) via:

```
git submodule update --remote --init --recursive
```

### Testing the ResMiCo package (optional)

`pytest -s`

### Installing the ResMiCo package into the conda environment

`python setup.py install`


# Usage

## ResMiCo-Sm

### Creating feature tables for genomes (MAGs)

If you just want to create feature tables so that you can run
`ResMiCo predict` for contig misassembly classification,
then keep reading this section. 

If you instead want to create simulated training/testing datasets,
then see `Creating custom train/test data from reference genomes` below.

**Input:**

* A table of reference genomes & metagenome samples
  * The table maps reference genomes to metagenomes from which they originate.
    * If MAGs created by binning, you can either combine metagenome samples, or map genomes to many metagenome samples 
  * Table format: `<Taxon>\t<Fasta>\t<Sample>\t<Read1>\t<Read2>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the genome (MAG) fasta file (uncompressed or gzip'ed)
     * "Sample" = the metagenome sample from which the genome originated
       * Note: the 'sample' can just be gDNA from a cultured isolate (not a metagenome)
     * "Read1" = Illumina Read1 for the sample
     * "Read2" = Illumina Read2 for the sample
* The snakemake config file (e.g., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters
  * Note: the same config is used for simulations and feature table creation

### Running locally 

`snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>`

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

#### Running on SGE cluster

The SGE profile in [Ley Lab snakemake profiles](https://github.com/leylabmpi/snakemake_profiles)
should work without any modifications (read the README).

You can use `./snakemake_sge.sh` for convenience

### Output

> Assuming output directory is `./output/`

* `./output/map/`
  * Metagenome assembly error ML features
* `./output/logs/`
  * Shell process log files (also see the SGE job log files)
* `./output/benchmarks/`
  * Job resource usage info

#### Features table

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

Run `ResMiCo train -h` to get a full description.

### Creating custom train/test data from reference genomes

This is useful for training ResMiCo (DL) with a custom
train/test dataset (e.g., just biome-specific taxa). 

**Input:**

* A table listing refernce genomes. Two possible formats:
  * **Either** Genome-accession: `<Taxon>\t<Accession>`
     * "Taxon" = the species/strain name
     * "Accession" = the NCBI genbank genome accession 
     * The genomes will be downloaded based on the accession
  * **Or** Genome-fasta: `<Taxon>\t<Fasta>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the fasta of the genome sequence
     * Use this option if you already have the genome fasta files (uncompressed or gzip'ed)
* The snakemake config file (e.g., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters

Note: the column order for the tables doesn't matter, but the column names must be exact.

#### Output

The output will the be same as for feature generation, but with extra directories:

* `./output/genomes/`
  * Reference genomes
* `./output/MGSIM/`
  * Simulated metagenomes
* `./output/assembly/`
  * Metagenome assemblies
* `./output/true_errors/`
  * Metagenome assembly errors determined by using the references


## ResMiCo (DL)

Main interface: `ResMiCo -h`

Note: `ResMiCo` can be run without GPUs, but it will be much slower.

### Predicting with existing model

See `ResMiCo predict -h` 

### Training a new model

See `ResMiCo train -h` 

### Evaluating a model

See `ResMiCo evalulate -h`


