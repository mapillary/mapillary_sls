from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np

def collate_tuples(batch):
	# reshape input [b, N, C, W, H] => [b*N, C, W, H]
	# where N = len([-1, 1, [0]*neg]
	print(batch.shape)
	return torch.cat([batch[i][0] for i in range(len(batch))]), torch.cat([batch[i][1] for i in range(len(batch))])
