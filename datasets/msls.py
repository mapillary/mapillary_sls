import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
from .generic_dataset import ImagesFromList
from tqdm import tqdm


class MSLS(Dataset):
    def __init__(self, root_dir, cities = '', nNeg = 5, transform = None, mode = 'train', subtask = 'all', posDistThr = 10, negDistThr = 25, cached_queries = 1000, cached_negatives = 1000, positive_sampling = True):

        # initializing
        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        self.sideways = []
        self.night = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = 0.1
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.exclude_panos = True
        self.mode = mode
        self.subtask = subtask

        # other
        self.transform = transform

        # load data
        self.cities = cities.split(',')
        for city in self.cities:
            print("=====> {}".format(city))

            # get len of images from cities so far for indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)

            # load query data
            qData = pd.read_csv(join(root_dir, city, 'query', 'postprocessed.csv'), index_col = 0)
            qDataRaw = pd.read_csv(join(root_dir, city, 'query', 'raw.csv'), index_col = 0)

            # load database data
            dbData = pd.read_csv(join(root_dir, city, 'database', 'postprocessed.csv'), index_col = 0)
            dbDataRaw = pd.read_csv(join(root_dir, city, 'database', 'raw.csv'), index_col = 0)

            # filter based on panorama data
            if self.exclude_panos:
                qData = qData[(qDataRaw['pano'] == False).values]
                dbData = dbData[(dbDataRaw['pano'] == False).values]

            # filter based on subtasks
            if self.mode in ['test', 'val']:
                qIdx = pd.read_csv(join(root_dir, city, 'query', 'subtask_index.csv'), index_col = 0)
                dbIdx = pd.read_csv(join(root_dir, city, 'database', 'subtask_index.csv'), index_col = 0)

                if self.exclude_panos:
                    qIdx = qIdx[(qDataRaw['pano'] == False).values]
                    dbIdx = dbIdx[(dbDataRaw['pano'] == False).values]

                qData = qData[(qIdx[self.subtask] == True).values]
                dbData = dbData[(dbIdx[self.subtask] == True).values]

            # save full path for images
            self.qImages.extend([join(root_dir, city, 'query', 'images', key + '.jpg') for key in qData['key']])
            self.dbImages.extend([join(root_dir, city, 'database','images', key + '.jpg') for key in dbData['key']])

            # cast utm coordinates to work with faiss
            utmQ = qData[['easting', 'northing']].values.reshape(-1,2)
            utmDb = dbData[['easting', 'northing']].values.reshape(-1,2)

            # find positive images for training
            neigh = NearestNeighbors(algorithm = 'brute')
            neigh.fit(utmDb)
            D, I = neigh.radius_neighbors(utmQ, self.posDistThr)

            if mode == 'train':
                nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

            night, sideways = qData['night'].values, (qData['view_direction'] == 'Sideways').values
            for qidx in range(len(utmQ)):

                pidx = I[qidx]
                # the query image has at least one positive
                if len(pidx) > 0:

                    self.pIdx.append(pidx + _lenDb)
                    self.qIdx.append(qidx + _lenQ)

                    # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                    if self.mode == 'train':

                        self.nonNegIdx.append(nI[qidx] + _lenDb)

                        # gather meta which is useful for positive sampling
                        if night[qidx]: self.night.append(len(self.qIdx)-1)
                        if sideways[qidx]: self.sideways.append(len(self.qIdx)-1)

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx)
        self.dbImages = np.asarray(self.dbImages)
        self.sideways = np.asarray(self.sideways)
        self.night = np.asarray(self.night)

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threads = 8
        self.bs = 24

        if mode == 'train':

            # for now always 1-1 lookup.
            self.negCache = np.asarray([np.empty((0,), dtype=int)]*len(self.qIdx))

            # calculate weights for positive sampling
            if positive_sampling:
                self.__calcSamplingWeights__()
            else:
                self.weights = np.ones(len(self.qIdx)) / float(len(self.qIdx))

    def __calcSamplingWeights__(self):

        # length of query
        N = len(self.qIdx)

        # initialize weights
        self.weights = np.ones(N)

        # weight higher if from night or sideways facing
        if len(self.night) != 0:
            self.weights[self.night] += N / len(self.night)
        if len(self.sideways) != 0:
            self.weights[self.sideways] += N / len(self.sideways)

        # print weight information
        print("#Sideways [{}/{}]; #Night; [{}/{}]".format(len(self.sideways), N, len(self.night), N))
        print("Forward and Day weighted with {:.4f}".format(1))
        if len(self.night) != 0:
            print("Forward and Night weighted with {:.4f}".format(1 + N/len(self.night)))
        if len(self.sideways) != 0:
            print("Sideways and Day weighted with {:.4f}".format( 1 + N/len(self.sideways)))
        if len(self.sideways) != 0 and len(self.night) != 0:
            print("Sideways and Night weighted with {:.4f}".format(1 + N/len(self.night) + N/len(self.sideways)))

    def __load_image_data__(self, idx):

        img = Image.open(self.imlist[idx] + '.jpg')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indicies = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net):

        # reset triplets
        self.triplets = []

        # take n query images
        qidxs = np.asarray(self.subcache_indicies[self.current_subset])

        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform),**opt)
        ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.transform),**opt)
        nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.transform),**opt)
        
        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), net.meta['outputdim']).to(self.device)
            pvecs = torch.zeros(len(pidxs), net.meta['outputdim']).to(self.device)
            nvecs = torch.zeros(len(nidxs), net.meta['outputdim']).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc = 'compute query descriptors'):
                X, y = batch
                qvecs[i*bs:(i+1)*bs, : ] = net(X.to(self.device)).data
            for i, batch in tqdm(enumerate(ploader), desc = 'compute positive descriptors'):
                X, y = batch
                pvecs[i*bs:(i+1)*bs, :] = net(X.to(self.device)).data
            for i, batch in tqdm(enumerate(nloader), desc = 'compute negative descriptors'):
                X, y = batch
                nvecs[i*bs:(i+1)*bs, :] = net(X.to(self.device)).data

        print('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)
        
        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)
        
        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()
        
        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q,:], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]
            
            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg: continue
            
            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]
            
            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]
            
            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            target = [-1, 1] + [0]*len(hardestNeg)

            self.triplets.append((triplet, target))
        
        # increment subset counter
        self.current_subset += 1
        
    def __getitem__(self, idx):
        
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]
        
        # load images into triplet list
        output = [self.transform(Image.open(self.qImages[qidx]))]
        output.append(self.transform(Image.open(self.dbImages[pidx])))
        output.extend([self.transform(Image.open(self.dbImages[idx])) for idx in nidx])

        return torch.stack(output), torch.tensor(target)
