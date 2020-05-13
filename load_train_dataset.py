from datasets.msls import MSLS
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
# load dataset for training
#####

# positive are defined within a radius of 10 m
posDistThr = 10

# negatives are defined outside a radius of 25 m
negDistThr = 25

# number of negatives per triplet
nNeg = 5

# number of cached queries
cached_queries = 1000

# number of cached negatives
cached_negatives = 1000

# whether to use positive sampling
positive_sampling = True

# choose the cities to load
cities = TRAIN_CITIES[:2]


train_dataset = MSLS(root_dir, cities = cities, transform = transform, mode = 'train', 
					negDistThr = negDistThr, posDistThr = postDistThr, nNeg = nNeg, cached_queries = cached_queries,
					cached_negatives = cached_negatives, positive_sampling = positive_sampling)

# It requires a model to create the cache... We will provide the training script later on. 
# Without a model to create the cache this dataset is rather useless... 
