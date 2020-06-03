import argparse
from os.path import basename as bn
import urllib
import zipfile
from os.path import basename as bn
from pathlib import Path

import numpy as np

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import eval

def create_dummy_predictions(prediction_path, dataset):
    print("==> Prediction file doesn't exist")
    print("==> We create a new dummy prediction file at :")
    print("==> ", prediction_path)

    numQ = len(dataset.qIdx)
    numDb = len(dataset.dbImages)

    # all keys in the database and query keys
    query_keys = np.asarray([','.join([bn(k)[:-4] for k in key.split(',')]) for key in dataset.qImages[dataset.qIdx]]).reshape(numQ,1)
    database_keys = [','.join([bn(k)[:-4] for k in key.split(',')]) for key in dataset.dbImages]
    
    # choose n = min(5, numDb) random elements from the database
    ranks = np.asarray([np.random.choice(database_keys, replace=False, size = (min(5, numDb))) for q in range(numQ)])

    # save the dummy predictions
    first_row_str = ' '.join([str(i) for i in ranks[0][:-1]]) + ' and ' + str(ranks[0][-1]) + '.'
    np.savetxt(prediction_path, np.concatenate([query_keys, ranks], axis=1), fmt='%s',
               header="Each row contains the a query key followed by N prediction keys\n"
                      "The format is valid for all tasks (im2im, im2seq, ...).\n"
                      
                      "For example, in this file, the image/sequence with key " + query_keys[0,0] + 
                      " is predicted to match with " + first_row_str)

def download_msls_sample(path):
    print("Downloading MSLS sample to {}".format(path))
    path.mkdir(parents=True)
    path_dl, _ = urllib.request.urlretrieve("https://static.mapillary.com/MSLS_samples.zip")
    with zipfile.ZipFile(path_dl, 'r') as zf:
        zf.extractall(path)

def main():

    parser = argparse.ArgumentParser()
    root_default = Path(__file__).parent / 'MSLS_sample'
    parser.add_argument('--prediction',
                        type=Path,
                        default='example_msls_prediction.csv',
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
                             ' Leave blank to use the default validation set (sf + cph)')
    parser.add_argument('--task',
                        type=str,
                        default='im2im',
                        help='Task to evaluate on: '
                             '[im2im, seq2im, im2seq, seq2seq]')
    parser.add_argument('--seq_length',
                        type=int,
                        default=1,
                        help='Sequence length to evaluate on')
    parser.add_argument('--subtask',
                        type=str,
                        default='all',
                        help='Subtask to evaluate on: '
                             '[all, s2w, w2s, o2n, n2o, d2n, n2d]')
    args = parser.parse_args()

    if not args.msls_root.exists():
        if args.msls_root == root_default:
            download_msls_sample(args.msls_root)
        else:
            raise FileNotFoundError("Not found: {}".format(args.msls_root))

    # select for which ks to evaluate
    ks = [1, 5, 10, 20]

    dataset = MSLS(args.msls_root, cities = args.cities, mode = 'val', posDistThr = args.threshold, 
                    task = args.task, seq_length = args.seq_length, subtask = args.subtask)

    # get query and positive image keys    
    positive_keys = [[','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages[pos]] for pos in dataset.pIdx]
    query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages[dataset.qIdx]]
    
    # create dummy predictions
    create_dummy_predictions(args.prediction, dataset)
    
    # load prediction rankings
    predictions = np.loadtxt(args.prediction, ndmin=2, dtype=str)

    # Ensure that there is a prediction for each query image
    for k in query_keys:
        assert k in predictions[:, 0], "You didn't provide any predictions for image {}".format(k)

    # Check if there are predictions that don't correspond to any query images
    for k in predictions[:, 0]:
        if k not in query_keys:
            print("Ignoring predictions for {}. It is not in the selected cities".format(k))
    predictions = np.array([l for l in predictions if l[0] in query_keys])

    # evaluate ranks
    metrics = eval(query_keys, positive_keys, predictions, ks=ks)

    # print metrics
    for metric in ['recall', 'map']:
        for i, k in enumerate(ks): 
            print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
        print()


if __name__ == "__main__":
    main()
