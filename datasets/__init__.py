from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np

def configure_dataset(opt, transform, mode = 'train'):

	if opt.dataset.lower() == 'msls':
		from datasets.msls import MSLS

		postDistThr = 10 if mode == 'train' else 25
		cities = opt.cities_train if mode == 'train' else opt.cities_test

		dataset = MSLS(opt.root_dir, cities = cities, nNeg = opt.num_neg, transform = transform,
						mode = mode, subtask = 'all', posDistThr = postDistThr, negDistThr = 25,
						cached_queries = opt.cached_queries, cached_negatives = opt.cached_negatives, positive_sampling = opt.positive_sampling)

	return dataset

def configure_transform(opt, meta):

	normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
	transform = transforms.Compose([
		transforms.Resize(opt.image_dim),
		transforms.ToTensor(),
		normalize,
	])

	return transform

def collate_tuples(batch):
	# reshape input [b, N, C, W, H] => [b*N, C, W, H]
	# where N = len([-1, 1, [0]*neg]

	return torch.cat([batch[i][0] for i in range(len(batch))]), torch.cat([batch[i][1] for i in range(len(batch))])

"""
class DataLoader:
	""multi-threaded data loading""

	def __init__(self, dataset, opt, mode):
		self.opt = opt

		shuffle = True if mode == 'train' else False
		self.batch_size = opt.bs_train if mode == 'train' else opt.bs_test
		self.collate_fn = collate_tuples if mode == 'train' else None
		self.dataset = dataset

		if self.opt.max_dataset_size < np.inf:
			self.dataset.qIdx = self.dataset.qIdx[:self.opt.max_dataset_size]
			self.dataset.pIdx = self.dataset.qIdx[:self.opt.max_dataset_size]

		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=self.batch_size,
			shuffle=shuffle,
			num_workers=int(opt.num_threads),
			pin_memory=True,
			collate_fn = self.collate_fn)

	def __len__(self):
		return min(len(self.dataset), self.opt.max_dataset_size)

	def __iter__(self):
		for i, data in enumerate(self.dataloader):
			
			if i * self.batch_size >= self.opt.max_dataset_size:
				break

			yield data

"""
