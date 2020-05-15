from datasets.msls import MSLS
from utils.evaluate import evaluate
import numpy as np
import os

def create_dummy_predictions(prediction_path, dataset):
    print("==> Prediction file doesn't exist")
    print("==> We create a new dummy prediction file at :")
    print("==> ", prediction_path)

    numQ = len(dataset.qIdx)
    numDb = len(dataset.dbImages)

    # all elements in the database
    database_elements = np.arange(0, numDb)

    # choose n = min(100, numDb) random elements from the database
    ranks = [np.random.choice(database_elements, replace=False, size = (min(100, numDb))) for q in range(numQ)]
    
    # save the dummy predictions
    np.savetxt(prediction_path, ranks, fmt='%d')

if __name__ == "__main__":
    
    # path to MSLS dataset
    root_dir = "../../data/MSLS_train_val"

    # path to evaluation file
    prediction_path = 'msls_prediction.csv'

    # choose the cities to load (comma separated)
    cities = "zurich"

    # select threshold to evaluate on
    posDistThr = 25

    # select for which ks to evaluate
    ks = [1, 5, 10, 20]

    # choose subtask to test on [all, s2w, w2s, o2n, n2o, d2n, n2d]
    subtask = 'all'

    # use val when GPS / UTM is available, and use test when these are not available
    dataset = MSLS(root_dir, cities = cities, mode = 'val', 
                        posDistThr = posDistThr, subtask = subtask)

    positive_indices = dataset.pIdx
    query_indices = dataset.qIdx
    
    if not os.path.exists(prediction_path):
        # no GPS / UTM available in test
        dataset = MSLS(root_dir, cities = cities, mode = 'test', 
                            posDistThr = posDistThr, subtask = subtask)

        # create a dummy example of a prediction file
        create_dummy_predictions(prediction_path, dataset)
        
    # load prediciton rankings
    predictions = np.loadtxt(prediction_path, ndmin=2)

    # only take query indices that are not panoramic
    predictions = predictions[query_indices, :]

    # evaluate ranks
    metrics = evaluate(predictions, positive_indices, ks=ks)

    # print metrics
    for metric in ['recall', 'map']:
        for i, k in enumerate(ks): 
            print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
        print()


