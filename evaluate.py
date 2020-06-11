import argparse
from os.path import basename as bn
from pathlib import Path

import numpy as np

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import eval, create_dummy_predictions, download_msls_sample


def main():

    parser = argparse.ArgumentParser()
    root_default = Path(__file__).parent / 'MSLS_sample'
    parser.add_argument('--prediction',
                        type=Path,
                        default=Path(__file__).parent / 'files' / 'example_msls_im2im_prediction.csv',
                        help='Path to the prediction to be evaluated')
    parser.add_argument('--msls-root',
                        type=Path,
                        default=root_default,
                        help='Path to MSLS containing the train_val and/or test directories')
    parser.add_argument('--threshold',
                        type=float,
                        default=25,
                        help='Positive distance threshold defining ground truth pairs')
    parser.add_argument('--cities',
                        type=str,
                        default='zurich',
                        help='Comma-separated list of cities to evaluate on.'
                             ' Leave blank to use the default validation set (sf,cph)')
    parser.add_argument('--task',
                        type=str,
                        default='im2im',
                        help='Task to evaluate on: '
                             '[im2im, seq2im, im2seq, seq2seq]')
    parser.add_argument('--seq-length',
                        type=int,
                        default=3,
                        help='Sequence length to evaluate on for seq2X and X2seq tasks')
    parser.add_argument('--subtask',
                        type=str,
                        default='all',
                        help='Subtask to evaluate on: '
                             '[all, s2w, w2s, o2n, n2o, d2n, n2d]')
    parser.add_argument('--output',
                        type=Path,
                        default=None,
                        help='Path to dump the metrics to')
    args = parser.parse_args()

    if not args.msls_root.exists():
        if args.msls_root == root_default:
            download_msls_sample(args.msls_root)
        else:
            print(args.msls_root, root_default)
            raise FileNotFoundError("Not found: {}".format(args.msls_root))

    # select for which ks to evaluate
    ks = [1, 5, 10, 20]
    if args.task == 'im2im' and args.seq_length > 1:
        print(f"Ignoring sequence length {args.seq_length} for the im2im task. (Setting to 1)")
        args.seq_length = 1

    dataset = MSLS(args.msls_root, cities = args.cities, mode = 'val', posDistThr = args.threshold, 
                    task = args.task, seq_length = args.seq_length, subtask = args.subtask)

    # get query and positive image keys  
    database_keys =  [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages]
    positive_keys = [[','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages[pos]] for pos in dataset.pIdx]
    query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages[dataset.qIdx]]
    
    # create dummy predictions
    if not args.prediction.exists():
        create_dummy_predictions(args.prediction, dataset)
    
    # load prediction rankings
    predictions = np.loadtxt(args.prediction, ndmin=2, dtype=str)

    # Ensure that there is a prediction for each query image
    for k in query_keys:
        assert k in predictions[:, 0], "You didn't provide any predictions for image {}".format(k)

    # Ensure that all predictions are in database
    for k in predictions[:, 1:]:
        assert np.in1d(k, database_keys).all(), "Some of your predictions are not in the database for the selected task {}".format(k)

    # Ensure that all predictions are unique
    for k in range(len(query_keys)):
        assert len(predictions[k, 1:]) == len(np.unique(predictions[k, 1:])), "You have douplicate predictions for image {}".format(query_keys[k])

    # Ensure that all queries only exists once
    for k in predictions[:,0]:
        assert sum(k == predictions[:, 0]) == 1

    assert len(predictions[:,0]) == len(np.unique(predictions[:,0])), "You have douplicate query images"

    # Check if there are predictions that don't correspond to any query images
    for k in predictions[:, 0]:
        if k not in query_keys:
            print("Ignoring predictions for {}. It is not in the selected cities".format(k))
    predictions = np.array([l for l in predictions if l[0] in query_keys])

    # evaluate ranks
    metrics = eval(query_keys, positive_keys, predictions, ks=ks)

    f = open(args.output, 'a') if args.output else None
    # save metrics
    for metric in ['recall', 'map']:
        for k in ks:
            line =  '{}_{}@{}: {:.3f}'.format(args.subtask,
                                              metric,
                                              k,
                                              metrics['{}@{}'.format(metric, k)])
            print(line)
            if f:
                f.write(line + '\n')
    if f:
        f.close()

if __name__ == "__main__":
    main()
