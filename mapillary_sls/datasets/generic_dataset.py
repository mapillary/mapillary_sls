import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random

class ImagesFromList(Dataset):
	def __init__(self, images, transform):

	    self.images = np.asarray(images)
	    self.transform = transform

	def __len__(self):
	    return len(self.images)

	def __getitem__(self, idx):

		img = Image.open(self.images[idx])
		img = self.transform(img)

		return img, idx


