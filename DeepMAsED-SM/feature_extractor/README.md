## Building
### Prerequisites
  - a relatively new CLang or GCC (e.g. gcc-8) (C++17 support required)
  - cmake 3.13 or newer
  - zlib (it's just there on most systems)
### Building
```
cd bam2feat
mkdir build; cd build
cmake ..
make -j
```  
The binary is called `bam2feat` and will be in the `build` directory. This is what you have to run instead of
 `bam2feat.py`.
 
## Usage instructions
I kept as much as possible the interface to the old `bam2feat.py` untouched, so this should be a drop-in replacement
 for `bam2feat.py`, with a few small tweaks
   
  - for speed reasons, the BAM files must be indexed (so that one can easily find the reads for a specific contig)
  - for speed reasons, bam2feat doesn't write to the console; it needs an output file specified via the `--o
   <output_file>` flag
  - bam2feat writes directly to a gzipped stream; no need to gzip the result anymore
  
## Example of usage:
```
  ./bam2feat --bam_file chr20.bam --fasta_file GRCh37.p13.genome.fa --o ~/tmp/resmico --procs 4
```
