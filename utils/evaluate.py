import torch
import numpy as np


def rank_embeddings(qvecs, dbvecs):

	# search and rank
	scores = torch.mm(qvecs, dbvecs.T)
	ranks = torch.argsort(-scores, axis=1)

	ranks = ranks.cpu().numpy()

	return ranks

def evaluate(query_keys, positive_keys, predictions, ks = [1,5,10,20]):
	
	# ensure that the positive and predictions are for the same elements
	pred_queries = predictions[:,0:1]
	pred2gt = [np.where(query_keys == pred_key)[0][0] for pred_key in pred_queries]

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

	metrics = evaluate(qkeys, pkeys, predictions, ks = ks)

	# print metrics
	for metric in ['recall', 'map']:
		for i, k in enumerate(ks): 
			print('{}@{} = {:.3f}'.format(metric, k, metrics['{}@{}'.format(metric, k)]))
		print()




