import os
import bz2
import gzip
import logging
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
    header = dict()
    predictions = dict()
    with _open(infile) as inF:
        for i,line in enumerate(inF):
            line = _decode(line).rstrip().split(delim)
            # header
            if i == 0:
                header = {x:ii for ii,x in enumerate(line)}
                for col in ['cont_name', 'score']:
                    try:
                        _ = header[col]
                    except KeyError:
                        raise KeyError(f'Cannot find required column: "{col}"')
                continue
            # body
            contig_id = os.path.split(line[header['cont_name']])[1]
            score = float(line[header['score']])
            predictions[contig_id] = score
    return predictions

def filter_fasta(fasta_file, predictions, pred_score_cutoff, add_score=False, ignore_missing=False):
    """
    Filtering fasta based on contig prediction scores
    """
    logging.info('Filtering fasta...')
    contig_id = None
    pred_score = None
    stats = {'kept':0, 'filtered':0, 'total':0}
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
                    pred_score = float(predictions[contig_id])
                except KeyError:
                    msg = 'Cannot find contig "{}" in the list of predictions'
                    if ignore_missing is True:
                        logging.warning('WARNING: ' + msg.format(contig_id))
                    else:
                        raise KeyError(msg.format(contig_id))
                # status
                if pred_score is not None and pred_score < pred_score_cutoff:
                    stats['kept'] += 1
                else:
                    stats['filtered'] += 1
                # add score to sequence header?
                if add_score is True:
                    line = '>{} score={}'.format(contig_id, round(pred_score,6))
            # applying prediction score cutoff
            if pred_score is not None and pred_score < pred_score_cutoff:
                print(line)
    # status
    logging.info('No. of input contigs: {}'.format(stats['total']))
    logging.info('No. of filtered contigs: {}'.format(stats['filtered']))
    logging.info('No. of output contigs: {}'.format(stats['kept']))

def score_cutoff_check(score_cutoff):
    if score_cutoff < 0:
        logging.warning('WARNING: --score-cutoff < 0')
    elif score_cutoff > 1:
        logging.warning('WARNING: --score-cutoff > 1')
    
def main(args):
    # checks
    score_cutoff_check(args.score_cutoff)
    # predictions
    predictions = parse_predictions(args.prediction_table, args.score_delim)
    # fasta filter
    filter_fasta(args.fasta_file, predictions, args.score_cutoff,
                 args.add_score, args.ignore_missing)

if __name__ == '__main__':
    pass
