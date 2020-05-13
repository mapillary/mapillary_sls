from datasets.msls import MSLS
from datasets.generic_dataset import ImagesFromList
from utils.utils import configure_transform

VAL_CITIES = ['cph', 'sf']
TRAIN_CITIES = ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki", \
				"tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok", \
				"budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"]
TEST_CITES ["miami","athens","buenosaires","stockholm","bengaluru","kampala"]


ROOT_DIR = "" # INSERT ROOT_DIR

# get transform
meta = {'mean': [], 'std': []}
transform = configure_transform(image_dim = 640, meta = meta)

#####
# load dataset for testing
#####

# positive are defined within a radius of 25 m
posDistThr = 25

# choose the cities to load
cities = VAL_CITIES

# choose subtask to test on [all, s2w, w2s, o2n, n2o, d2n, n2d]
subtask = 'all'

val_dataset = MSLS(root_dir, cities = cities, transform = transform, mode = 'test', 
					subtask = subtask, posDistThr = posDistThr)

# get images
qLoader = DataLoader(ImagesFromList(self.val_dataset.qImages[self.val_dataset.qIdx], self.transform), **opt)
dbLoader = DataLoader(ImagesFromList(self.val_dataset.dbImages, self.transform), **opt)

# get positive index (we allow some more slack: default 25 m)
pIdx = self.val_dataset.pIdx

# Now you can get the index like a normal pytorch dataloader

