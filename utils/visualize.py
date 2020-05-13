import numpy as np
import matplotlib.pyplot as plt

def denormalize(im):
	image = im.numpy()
	im = (image - np.min(image)) / (np.max(image) - np.min(image))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im

def visualize_triplets(batch):

	images, labels = batch

	batch_count = 0
	for image, label in zip(images, labels):
		if batch_count > 5:
			break

		image = denormalize(image)

		if label == -1:
			batch_count += 1
			neg_count = 0
			plt.imshow(np.transpose(image,(1,2,0)))
			plt.title("batch {} => anchor".format(batch_count))
			plt.show()
		elif label == 1:
			plt.imshow(np.transpose(image,(1,2,0)))
			plt.title("batch {} => positive".format(batch_count))
			plt.show()
		elif label == 0:
			neg_count += 1
			plt.imshow(np.transpose(image,(1,2,0)))
			plt.title("batch {} => negative {}".format(batch_count, neg_count))
			plt.show()
