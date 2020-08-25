#  Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import matplotlib.pyplot as plt
import torch

def denormalize(im):
	image = im.numpy()
	im = (image - np.min(image)) / (np.max(image) - np.min(image))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im

def visualize_triplets(batch, task):

	sequences, labels = batch

	N = labels.shape[1]

	if task == 'im2im':
		q_seq_length, db_seq_length = 1,1
	elif task == 'seq2seq':
		seq_length = sequences.shape[1] // (N)
		q_seq_length, db_seq_length = seq_length, seq_length
	elif task == 'im2seq':
		seq_length = (sequences.shape[1] - 1) // (N - 1)
		q_seq_length, db_seq_length = 1, seq_length
	elif task == 'seq2im':
		seq_length = sequences.shape[1] - (N - 1)
		print(seq_length)
		q_seq_length, db_seq_length = seq_length, 1

	chuncks = list(np.concatenate([[q_seq_length], [db_seq_length]*(N - 1)]))

	for batch_idx in range(min(5, len(sequences))):
		seq_batch_split = torch.split(sequences[batch_idx], chuncks)
		for seq, label in zip(seq_batch_split, labels[batch_idx]):

			seq = [denormalize(im) for im in seq]

			if label == -1:
				neg_count = 0

				plt.figure(figsize=(15,5)) if q_seq_length > 1 else plt.figure(figsize=(5,5))
				for i in range(q_seq_length):
					plt.subplot(1, q_seq_length, i+1)
					plt.imshow(np.transpose(seq[i],(1,2,0)))
					plt.title("batch {} => anchor".format(batch_idx))
				plt.show()

			elif label == 1:
				plt.figure(figsize=(15,5)) if db_seq_length > 1 else plt.figure(figsize=(5,5))
				for i in range(db_seq_length):
					plt.subplot(1, db_seq_length, i+1)
					plt.imshow(np.transpose(seq[i],(1,2,0)))
					plt.title("batch {} => positive".format(batch_idx))
				plt.show()

			elif label == 0:
				neg_count += 1
				plt.figure(figsize=(15,5)) if db_seq_length > 1 else plt.figure(figsize=(5,5))
				for i in range(db_seq_length):
					plt.subplot(1, db_seq_length, i+1)
					plt.imshow(np.transpose(seq[i],(1,2,0)))
					plt.title("batch {} => negative {}".format(batch_idx, neg_count))
				plt.show()
