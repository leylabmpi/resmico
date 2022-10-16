import os
import bz2
import gzip
import shutil
import random
import logging
import multiprocessing as mp
from pkg_resources import resource_filename
from functools import partial
from distutils.spawn import find_executable
from subprocess import Popen, PIPE

import pysam

from resmico import utils


# functions
def _open(infile, mode='rb'):
    """Openning of input, regardless of compression."""
    if infile.endswith('.bz2'):
        return bz2.open(infile, mode)
    elif infile.endswith('.gz'):
        return gzip.open(infile, mode)
    else:
        return open(infile)


def _decode(x):
    """
    Decoding input, if needed
    """
    try:
        x = x.decode('utf-8')
    except AttributeError:
        pass
    return x


def file_exists(F, base_dir):
    """
    Does file exist as-is or at least with the appended base_dir?
    """
    if not os.path.exists(F):
        FF = os.path.join(base_dir, F)
        if not os.path.exists(FF):
            raise IOError(f'Cannot find file: "{F}" or "{FF}"')
        else:
            F = FF
    return F


def parse_input(infile):
    """Parsing tab-delimited input table."""
    base_dir = os.path.split(infile)[0]
    header = dict()
    idx = []
    with open(infile) as inF:
        for i, line in enumerate(inF):
            line = line.rstrip().split('\t')
            if line[0] == '':
                # blank line
                continue
            if len(line) < 4:
                raise ValueError('The input table must have >=4 columns!')
            if i == 0:
                # header
                header = {x.lower(): ii for ii, x in enumerate(line)}
                for k, v in {'Fasta': 'fasta', 'BAM': 'bam'}.items():
                    try:
                        _ = header[v]
                    except KeyError:
                        msg = f'Cannot find required column: "{k}"'
                        raise KeyError(msg)
                continue
            # body
            taxon = line[header['taxon']]
            sample = line[header['sample']]
            fasta = file_exists(line[header['fasta']], base_dir)
            bam = file_exists(line[header['bam']], base_dir)
            idx.append([bam, fasta, sample, taxon])
    return idx


def run_cmd(cmd):
    """
    Simple run of a command
    """
    cmd = [str(x) for x in cmd]
    logging.info('  CMD: {}'.format(' '.join(cmd)))
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    rc = p.returncode
    if rc != 0:
        for line in output.decode().split('\n'):
            print(line)
        for line in err.decode().split('\n'):
            print(line)
        raise ValueError('Return code: {}'.format(rc))
    return rc


def uncomp_ref(fasta_file, tmpdir):
    """
    Uncompressing reference genome (if needed)
    """
    logging.info('  Uncompressing fasta...')
    outfile = os.path.join(tmpdir, os.path.split(fasta_file)[1])
    with _open(fasta_file) as inF, open(outfile, 'w') as outF:
        for line in inF:
            outF.write(_decode(line))
    return outfile


def faidx_ref(fasta_file, exe):
    """
    Indexing a reference via `samtools faidx`
    """
    logging.info('  Indexing fasta...')
    outfile = fasta_file + '.fai'
    cmd = [exe, 'faidx', fasta_file]
    run_cmd(cmd)
    return outfile


def get_basename(F):
    return os.path.splitext(os.path.split(F)[1])[0]


def index_bam(bam_file, exe, n_threads):
    """
    Indexing a bam file via `samtools index`
    """
    cmd = [exe, 'index', '-@', n_threads, bam_file]
    run_cmd(cmd)
    return bam_file + '.bai'


def sort_bam(bam_file, exe, tmpdir, n_threads):
    """
    Sorting a bam file via `samtools sort`
    """
    tmpdir = os.path.join(tmpdir, 'bam_sort_TMP')
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    outfile = os.path.splitext(bam_file)[0] + '_sorted.bam'
    cmd = [exe, 'sort', '-@', n_threads, '-o', outfile, bam_file]
    run_cmd(cmd)
    return outfile


def subsample_bam(bam_file, fasta_file, tmpdir, max_coverage, max_insert_size=30000):
    """
    Subsample the per-contig coverage from the bam file
    """
    bam = pysam.AlignmentFile(bam_file)
    fasta = pysam.FastaFile(fasta_file)
    outfile = os.path.join(tmpdir, get_basename(bam_file) + '_sub.bam')
    output = pysam.AlignmentFile(outfile, 'wb', template=bam)
    # getting contig lengths
    logging.info('  Subsampling reads...')
    contig_cov = {contig: 0 for contig in bam.references}
    for contig in bam.references:
        logging.info(f'    Processing contig: {contig}')
        contig_len = len(fasta.fetch(contig))
        for read in sorted(bam.fetch(contig), key=lambda k: random.random()):
            # new cov
            added_cov = read.query_length / contig_len
            # if max cov hit, skipping read
            if contig_cov[read.reference_name] + added_cov >= max_coverage:
                continue
            # if very large insert size, skipping
            if abs(read.template_length) > max_insert_size:
                continue
            # keeping read & tracking added coverage
            contig_cov[read.reference_name] = contig_cov[read.reference_name] + added_cov
            output.write(read)
        mean_cov = round(contig_cov[contig], 2)
        logging.info(f'    Final mean coverage: {mean_cov}')
    return outfile


def bam2feat(bam_file, fasta_file, outdir, exe, args):
    """
    Calling bam2feat on bam + fasta
    """
    cmd = [exe, '--procs', args.n_threads, '-queue_size', args.queue_size,
           '--window', args.window, '-breakpoint_margin', args.breakpoint_margin,
           '--o', outdir, '--bam_file', bam_file, '--fasta_file', fasta_file]
    run_cmd(cmd)
    return outdir


def _run_bam2feat(x, exe, outdir, tmpdir, args):
    """
    Per-BAM processing of input files and running bam2feat
    """
    logging.info(f'BAM file: {x[2]} => {x[0]}')
    logging.info(f'Fasta file: {x[3]} => {x[1]}')
    # tmpdir
    tmpdir = os.path.join(tmpdir, x[2], x[3])
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    logging.info(f'Temp Dir: {tmpdir}')
    # uncompress reference
    ref_tmp = uncomp_ref(x[1], tmpdir)
    # index ref
    ref_tmp_faidx = faidx_ref(ref_tmp, exe['samtools'])
    # index bam (if needed)
    bam_bai = x[0] + '.bai'
    if not os.path.isfile(bam_bai):
        bam_bai = index_bam(x[0], exe['samtools'], args.n_threads)
    # subsample bam
    bam_sub = subsample_bam(x[0], ref_tmp, tmpdir, args.max_coverage)
    # sort bam
    bam_sub = sort_bam(bam_sub, exe['samtools'], tmpdir, args.n_threads)
    # index bam
    bam_bai = index_bam(bam_sub, exe['samtools'], args.n_threads)
    # bam2feat
    if args.outdir_flat:
        outdir_feat = outdir
    else:
        outdir_feat = os.path.join(outdir, x[2], x[3])
    if not os.path.isdir(outdir_feat):
        os.makedirs(outdir_feat)
    logging.info(f'Outdir: {outdir_feat}')
    bam2feat(bam_sub, ref_tmp, outdir_feat, exe['bam2feat'], args)
    # clean up
    logging.info(f'Cleaning up Temp Dir: {tmpdir}')
    shutil.rmtree(tmpdir, ignore_errors=True)
    return [x[3], x[2], outdir_feat]


def run_bam2feat(bam_fasta, exe, args):
    """
    Main pipeline interface for bam2feat, including input file processing
    """
    # seed
    random.seed(args.seed)
    # outdir
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    # per-BAM processing
    func = partial(_run_bam2feat, exe=exe,
                   outdir=args.outdir,
                   tmpdir=args.tmpdir,
                   args=args)
    if args.n_proc < 2:
        logging.info('Processing input...')
        res = map(func, bam_fasta)
    else:
        logging.info('Processing input in parallel...')
        set_logger(logging.WARNING)
        pool = mp.Pool(args.n_proc)
        res = pool.map(func, bam_fasta)
    return [x for x in res]


def write_feat_table(feat_files, outdir):
    """
    Writing tab-delim feature file table 
    """
    outfile = os.path.join(outdir, 'feature_files.tsv')
    with open(outfile, 'w') as outF:
        header = ['Taxon', 'Sample', 'feature_file']
        outF.write('\t'.join(header) + '\n')
        for line in feat_files:
            outF.write('\t'.join(line) + '\n')
    logging.info(f'Feature file table written: {outfile}')


def main(args):
    logging.info('Starting bam2feat...')
    # check that bam2feat is available
    exe = {'bam2feat': resource_filename('resmico', 'bam2feat'),
           'samtools': 'samtools'}
    for k, v in exe.items():
        if find_executable(v) is None:
            raise OSError(f'Cannot find executable: {k}')
    # parse input table
    logging.info(f'Parsing input table: {args.input_table}')
    bam_fasta = parse_input(args.input_table)
    # run bam2feat in parallel
    feat_files = run_bam2feat(bam_fasta, exe, args)
    # writing table
    write_feat_table(feat_files, args.outdir)
    # clean up
    shutil.rmtree(args.tmpdir, ignore_errors=True)


if __name__ == '__main__':
    pass
