from datasets.msls import MSLS
from utils.evaluate import evaluate
import numpy as np
from pathlib import Path
import argparse
from os.path import basename as bn

def create_dummy_predictions(prediction_path, dataset):
    print("==> Prediction file doesn't exist")
    print("==> We create a new dummy prediction file at :")
    print("==> ", prediction_path)

    numQ = len(dataset.qIdx)
    numDb = len(dataset.dbImages)

    # all keys in the database and query keys
    query_keys = np.asarray([bn(key)[:-4] for key in dataset.qImages[dataset.qIdx]]).reshape(numQ,1)
    database_keys = [bn(key)[:-4] for key in dataset.dbImages]
    
    # choose n = min(5, numDb) random elements from the database
    ranks = np.asarray([np.random.choice(database_keys, replace=False, size = (min(5, numDb))) for q in range(numQ)])

    # save the dummy predictions
    first_row_str = ' '.join([str(i) for i in ranks[0][:-1]]) + ' and ' + str(ranks[0][-1]) + '.'
    np.savetxt(prediction_path, np.concatenate([query_keys, ranks], axis=1), fmt='%s',
               header="Each row contains the a query key followed by N prediction keys\n"
                      "The format is valid for all tasks (im2im, im2seq, ...).\n"
                      
                      "For example, in this file, the image/sequence with key " + query_keys[0,0] + 
                      " is predicted to match with " + first_row_str)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root_default = Path(__file__).parent / 'data'
    parser.add_argument('prediction',
                        type=Path,
                        default=Path(__file__).parent / 'msls_prediction.csv',
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
                             ' Leave blank to use all the validation set')
    parser.add_argument('--subtask',
                        type=str,
                        default='all',
                        help='Subtask to evaluate on: '
                             '[all, s2w, w2s, o2n, n2o, d2n, n2d]')
    args = parser.parse_args()

    # select for which ks to evaluate
    ks = [1, 5, 10, 20]

    dataset = MSLS(args.msls_root, cities = args.cities, mode = 'val',
                        posDistThr = args.threshold, subtask = args.subtask)

    # get query and positive image keys
    positive_keys = [[bn(key)[:-4] for key in dataset.dbImages[pos]] for pos in dataset.pIdx]
    query_keys = [bn(key)[:-4] for key in dataset.qImages[dataset.qIdx]]

    # create dummy predictions
    create_dummy_predictions(args.prediction, dataset)
    
    # load prediction rankings
    predictions = np.loadtxt(args.prediction, ndmin=2, dtype=str)
    
    # evaluate ranks
    metrics = evaluate(query_keys, positive_keys, predictions, ks=ks)

    # print metrics
    for metric in ['recall', 'map']:
        for i, k in enumerate(ks): 
            print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
        print()


