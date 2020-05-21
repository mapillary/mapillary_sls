from datasets.msls import MSLS
from utils.evaluate import evaluate
import numpy as np
from pathlib import Path
import argparse

def create_dummy_predictions(prediction_path, dataset):
    print("==> Prediction file doesn't exist")
    print("==> We create a new dummy prediction file at :")
    print("==> ", prediction_path)

    numQ = len(dataset.qIdx)
    numDb = len(dataset.dbImages)

    # all elements in the database
    database_elements = np.arange(0, numDb)

    # choose n = min(100, numDb) random elements from the database
    ranks = [np.random.choice(database_elements, replace=False, size = (min(5, numDb))) for q in range(numQ)]
    
    # save the dummy predictions
    first_row_str = ' '.join([str(i) for i in ranks[0][:-1]]) + ' and ' + str(ranks[0][-1]) + '.'
    np.savetxt(prediction_path, ranks, fmt='%d',
               header="The row number is the query ID.\n"
                      "Each row contains the predicted IDs for that query ID\n"
                      "The format is valid for all tasks (im2im, im2seq, ...).\n"
                      
                      "For example, in this file, the image/sequence with ID 0 is predicted to match with "
                      + first_row_str)

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

    positive_indices = dataset.pIdx
    query_indices = dataset.qIdx

    # load prediction rankings
    predictions = np.loadtxt(args.prediction, ndmin=2)

    predictions = predictions[query_indices, :]

    # evaluate ranks
    metrics = evaluate(predictions, positive_indices, ks=ks)

    # print metrics
    for metric in ['recall', 'map']:
        for i, k in enumerate(ks): 
            print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
        print()


