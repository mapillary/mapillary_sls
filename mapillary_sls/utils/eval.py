#  Copyright (c) Facebook, Inc. and its affiliates.

import urllib
import zipfile
from os.path import basename as bn

import numpy as np


def rank_embeddings(qvecs, dbvecs):

	# search and rank
	scores = np.matmul(qvecs, dbvecs.T)
	ranks = np.argsort(-scores, axis=1)

	return ranks

def eval(query_keys, positive_keys, predictions, ks = [1, 5, 10, 20]):

    # ensure that the positive and predictions are for the same elements
    pred_queries = predictions[:,0:1]
    pred2gt = [np.where(pred_queries == key)[0][0] for key in query_keys]

    # change order to fit with ground truth
    predictions = predictions[pred2gt,1:]

    recall_at_k = recall(predictions, positive_keys, ks)

    metrics = {}
    for i, k in enumerate(ks):
        metrics['recall@{}'.format(k)] = recall_at_k[i]
        metrics['map@{}'.format(k)] = mapk(predictions, positive_keys, k)

    return metrics

def recall(ranks, pidx, ks):

	recall_at_k = np.zeros(len(ks))
	for qidx in range(ranks.shape[0]):

		for i, k in enumerate(ks):
			if np.sum(np.in1d(ranks[qidx,:k], pidx[qidx])) > 0:
				recall_at_k[i:] += 1
				break

	recall_at_k /= ranks.shape[0]

	return recall_at_k

def apk(pidx, rank, k):
    if len(rank)>k:
        rank = rank[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(pidx), k)

def mapk(ranks, pidxs, k):

    return np.mean([apk(a,p,k) for a,p in zip(pidxs, ranks)])

if __name__ == "__main__":

	# predicted rankings
	predictions = np.asarray([["Q1", "DB0", "DB1", "DB2", "DB3", "DB4", "BD5"],
			 			["Q2", "DB0", "DB5", "DB4", "DB1", "DB3", "DB2"]])

	# query keys
	qkeys = np.asarray(["Q1", "Q2"])

	# ground truth rankings (positive idx)
	pkeys = np.asarray([["DB4","DB5"],
						["DB0","DB1"]])

	# evaluate at ks
	ks = [1, 2, 3]

	metrics = eval(qkeys, pkeys, predictions, ks = ks)

	# print metrics
	for metric in ['recall', 'map']:
		for i, k in enumerate(ks):
			print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
		print()


def create_dummy_predictions(prediction_path, dataset):
    print("==> Prediction file {} doesn't exist".format(prediction_path))
    print("==> We create a new dummy prediction file at :")
    print("==> ", prediction_path)

    numQ = len(dataset.qIdx)
    numDb = len(dataset.dbImages)

    # all keys in the database and query keys
    query_keys = np.asarray([','.join([bn(k)[:-4] for k in key.split(',')]) for key in dataset.qImages[dataset.qIdx]]).reshape(numQ,1)
    database_keys = [','.join([bn(k)[:-4] for k in key.split(',')]) for key in dataset.dbImages]
    # choose n = min(5, numDb) random elements from the database
    ranks = np.asarray([np.random.choice(database_keys, replace=False, size = (min(5, numDb))) for q in range(numQ)])

    if ',' in query_keys[0,0]:
        qtxt = f"sequence with keys {query_keys[0,0]}"
    else:
        qtxt = f"image with key {query_keys[0, 0]}"
    mtxt = ' '.join([str(i) for i in ranks[0][:-1]]) + ' and ' + str(ranks[0][-1]) + '.'
    if ',' in ranks[0,0]:
        mtxt = f"{len(ranks[0])} sequences: " + mtxt
    else:
        mtxt = f"{len(ranks[0])} images: " + mtxt
    hdr = ("Each row contains the key of a query image or a comma-separated set of keys for a query sequence,\n"
           "followed by space-separated predicted image keys or sets of comma-separated keys for predicted sequences\n"
           "The format is valid for all tasks (im2im, im2seq, seq2im, seq2seq)\n"
           "For example, in this file, the " + qtxt + " is predicted to match with " + mtxt)

    # save the dummy predictions
    np.savetxt(prediction_path, np.concatenate([query_keys, ranks], axis=1), fmt='%s',
               header=hdr)


def download_msls_sample(path):
    print("Downloading MSLS sample to {}".format(path))
    path.mkdir(parents=True)
    path_dl, _ = urllib.request.urlretrieve("https://static.mapillary.com/MSLS_samples.zip")
    with zipfile.ZipFile(path_dl, 'r') as zf:
        zf.extractall(path)
