import torch
import numpy as np


def rank_embeddings(qvecs, dbvecs):

	# search and rank
	scores = torch.mm(qvecs, dbvecs.T)
	ranks = torch.argsort(-scores, axis=1)

	ranks = ranks.cpu().numpy()

	return ranks

def evaluate(ranks, pidx, ks = [1,5,10,20]):

	recall_at_k = recall(ranks, pidx, ks)

	metrics = {}
	for i, k in enumerate(ks):
		metrics['recall@{}'.format(k)] = recall_at_k[i]
		metrics['map@{}'.format(k)] = mapk(ranks, pidx, k)

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





