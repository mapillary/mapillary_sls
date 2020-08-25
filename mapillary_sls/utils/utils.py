#  Copyright (c) Facebook, Inc. and its affiliates.

from torchvision import transforms
import torch

def configure_transform(image_dim, meta):

	normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
	transform = transforms.Compose([
		transforms.Resize(image_dim),
		transforms.ToTensor(),
		normalize,
	])

	return transform
