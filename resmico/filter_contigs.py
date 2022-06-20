import os
import re
import bz2
import gzip
import logging
import multiprocessing as mp
from functools import partial
from resmico import utils

def _open(infile, mode='rb'):
    """
    Openning of input, regardless of compression
    """
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

def parse_predictions(infile, delim=','):
    """
    Parsing contig prediction scores
    """
    logging.info('Parsing predictions table...')
    regex = re.compile(r'_CONTIG[0-9]+$')   
    header = dict()
    predictions = dict()
    with _open(infile) as inF:
        for i,line in enumerate(inF):
            line = _decode(line).rstrip().split(delim)
            # header
            if i == 0:
                header = {x:ii for ii,x in enumerate(line)}
                for col in ['cont_name', 'score', 'length']:
                    try:
                        _ = header[col]
                    except KeyError:
                        raise KeyError(f'Cannot find required column: "{col}"')
                continue
            # body
            contig_id = regex.sub('', os.path.split(line[header['cont_name']])[1])
            score = float(line[header['score']])
            contig_len = float(line[header['length']])
            predictions[contig_id] = [score, contig_len]
    # status
    n_obs = len(predictions.keys())
    logging.info(f'  No. of predictions: {n_obs}')
    scores = [x[0] for x in predictions.values()]
    min_score = round(min(scores),2)
    mean_score = round(sum(scores) / n_obs,2)
    max_score = round(max(scores),2)
    logging.info(f'  Min score: {min_score}; Mean score: {mean_score}; Max score: {max_score}')
    return predictions

def score_cutoff_check(score_cutoff):
    if score_cutoff < 0:
        logging.warning('WARNING: --score-cutoff < 0')
    elif score_cutoff > 1:
        logging.warning('WARNING: --score-cutoff > 1')

def parse_fasta_list(fasta_list):
    if len(fasta_list) > 1:
        return fasta_list
    fasta_files = []
    with open(fasta_list[0]) as inF:
        for line in inF:
            line = line.rstrip()
            if line == '':
                continue
            if line.startswith('>') or not os.path.exists(line):
                fasta_files = fasta_list
                break
            else:
                fasta_files.append(line)
    return fasta_files

def _filter_fasta(fasta_file, predictions, outdir, pred_score_cutoff, add_score=False,
                 error_on_missing=False, max_length=0):
    """
    Filtering fasta based on contig prediction scores
    """    
    logging.info(f'Filtering fasta: {fasta_file}')
    outfile = os.path.join(outdir, os.path.split(fasta_file)[1])
    if outfile == fasta_file:
        msg = 'Input path cannot equal output path: {} <=> {}'
        raise ValueError(msg.format(fasta_file, outfile))
    gzip_out = outfile.endswith('.gz')        
    contig_id = None
    pred_score = None
    contig_len = None
    stats = {'kept':0, 'filtered':0, 'total':0, 'match':0}
    if gzip_out:
        outF = gzip.open(outfile, 'wb')
    else:
        outF = open(outfile, 'w')
    with _open(fasta_file) as inF:
        for i,line in enumerate(inF):
            line = _decode(line).rstrip()
            if line == '':
                continue
            # seq header
            if line.startswith('>'):
                stats['total'] += 1
                contig_id = line.lstrip('>')
                # getting score for contig
                try:
                    pred_score = float(predictions[contig_id][0])
                    contig_len = float(predictions[contig_id][1])
                    stats['match'] += 1
                except KeyError:
                    msg = 'Cannot find contig "{}" in the list of predictions'
                    if error_on_missing is True:
                        raise ValueError(msg.format(contig_id))
                    pred_score = 0
                # status
                if max_length > 0 and contig_len > max_length:
                    # retaining long contigs, regardless of score
                    stats['kept'] += 1
                    pred_score = 0
                elif pred_score is not None and pred_score < pred_score_cutoff:
                    # retaining contig due to low score
                    stats['kept'] += 1
                else:
                    # score hits cutoff; filtering
                    stats['filtered'] += 1
                    continue
                # add score to sequence header?
                if add_score is True:
                    line = '>{} score={}'.format(contig_id, round(pred_score,6))
            # applying prediction score cutoff
            if pred_score is not None and pred_score < pred_score_cutoff:
                line = line + '\n'
                if gzip_out:
                    outF.write(line.encode('utf-8'))
                else:
                    outF.write(line)    
    # status
    outF.close()
    logging.info('  No. of input contigs: {}'.format(stats['total']))
    logging.info('  No. of contigs with a prediction: {}'.format(stats['match']))
    logging.info('  No. of filtered contigs: {}'.format(stats['filtered']))
    logging.info('  No. of output contigs: {}'.format(stats['kept']))
    logging.info(f'  File written: {outfile}')

def set_logger(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
        
def filter_fasta(fasta_files, predictions, outdir, pred_score_cutoff,
                 add_score=False, error_on_missing=False, max_length=0, n_proc=1):
    """
    Filter >=1 fasta based on prediction scores
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    func = partial(_filter_fasta,
                   predictions = predictions,
                   outdir = outdir,
                   pred_score_cutoff = pred_score_cutoff,
                   add_score = add_score,
                   error_on_missing = error_on_missing,
                   max_length = max_length)
    if n_proc < 2:
        res = map(func, fasta_files)
    else:
        logging.info('Filtering fasta files in parallel...')
        set_logger(logging.WARNING)
        pool = mp.Pool(n_proc)
        res = pool.map(func, fasta_files)
        set_logger(logging.INFO)
    return [x for x in res]
    
    
def main(args):
    # fasta file(s)
    args.fasta = parse_fasta_list(args.fasta)
    logging.info('No. of fasta files: {}'.format(len(args.fasta)))
    # checks
    score_cutoff_check(args.score_cutoff)
    # predictions
    predictions = parse_predictions(args.prediction_table, args.score_delim)
    # fasta filter
    filter_fasta(args.fasta, predictions, args.outdir, args.score_cutoff,
                 args.add_score, args.error_on_missing, args.max_length,
                 args.n_proc)

if __name__ == '__main__':
    pass
