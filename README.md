# Mapillary Street-level Sequences

## :newspaper: News

- *2020-12-30* - Added information about evaluation server
- *2020-07-14* - Released patch v1.1 fixing some corrupt images

## Description

Mapillary Street-level Sequences (MSLS) is a large-scale long-term place recognition dataset that contains 1.6M street-level images.

- ⬇️ Download: https://www.mapillary.com/dataset/places (sample [here](https://static.mapillary.com/MSLS_samples.zip))
- 📄 Paper: https://research.mapillary.com/publication/cvpr20c
- ️🧑‍⚖️ [Code of Conduct](CODE_OF_CONDUCT.md)
- 🗳️ [Contributing / Pull Requests](CONTRIBUTING.md)


## 🔥 Using MSLS

We've included an implementation of a PyTorch Dataset in [datasets/msls.py](mapillary_sls/datasets/msls.py).
It can be used for evaluation (returning database and query images) or for training (returning triplets).
Check out the [demo](demo.ipynb) to understand its usage.


#### 📊 Standalone evaluation script

A [standalone evaluation script](evaluate.py) is available for all tasks. It reads the predictions from a text file ([example](files/example_msls_im2im_prediction.csv)) and prints the metrics.

Here we show results of models consisting of a Resnet50 backbone followed by Generalized Mean Layer. The models are trained with either the standard triplet loss or the uncertainty-aware Bayesian triplet loss. All models are trained with standard hard negative mining on image resolution 224x224.

Results on test set (Miami, Athens, Buenos Aires, Stockholm, Bengaluru, Kampala):

|   Loss   |   R@1  |   R@5  |   R@10  |   R@20  |   M@1  |   M@5  |   M@10  |   M@20  |
|-|-|-|-|-|-|-|-|-|
|   Triplet Loss  | 0.372  |   0.522  |   0.582  |   0.636  |   0.372  |   0.261  |   0.234  |   0.228 |
|   Bayesian Triplet Loss  | 0.366	| 0.513	| 0.574	| 0.629 |	0.366|	0.253|	0.229|	0.222 |

Results on validation set (San Francisco, Copenhagen)

|   Loss   |   R@1  |   R@5  |   R@10  |   R@20  |   M@1  |   M@5  |   M@10  |   M@20  |
|-|-|-|-|-|-|-|-|-|
|   Triplet Loss  | 0.623  |   0.780  |   0.830  |   0.859  |   0.623  |   0.432  |   0.380  |   0.372 |
|   Bayesian Triplet Loss  | 0.618	| 0.746	| 0.805	| 0.839 |	0.618|	0.419|	0.369|	0.360 |

#### 📊 Evaluate on the test set

For evaluating on the test set, we've hosted a Codalab competition [here](https://codalab.lisn.upsaclay.fr/competitions/865)

Make sure to submit a single .zip file containing a single .csv file following the format described above for the standalone evaluation script. The ([example](files/example_msls_im2im_prediction.csv)) in this repository is for the validation set, you must make sure to submit predictions for the test set or your submission will fail.

If your submission fails, read the log to find out why. Known reasons:

- Worker out of space ("no space left on device" error): Please complain here and try again later.
- You didn't provide any predictions for image "X": Make sure your are submitting predictions for the test set.

If you get a different error, please file a Github Issue

## 📦 Package structure

- `images_vol_X.zip`: images, split into 6 parts for easier download.
- `metadata.zip`: a single zip archive containing the metadata.
- `patch_vX.Y.zip`: unzip any patches on top of the dataset to upgrade.

All the archives can be extracted in the same directory resulting in the following tree:

- train_val
    - `city`
        - query / database
            - images/`key`.jpg
            - seq_info.csv
            - subtask_index.csv
            - raw.csv
            - postprocessed.csv
- test
    - `city`
        - query / database
            - images/`key`.jpg
            - seq_info.csv
            - subtask_index.csv

The meta files include the following information:

- **raw.csv**: raw data recorded during capture
	- key
	- lon
	- lat
	- ca
	- captured_at
	- pano

- **seq_info.csv**: Sequence information
	- key
	- sequence_id
	- frame_number

- **postprocessed.csv**: Data derived from the raw images and metadata
	- key
	- utm (easting and northing)
	- night
	- control_panel
	- view_direction (Forward, Backward, Sideways)
	- unique_cluster

- **subtask_index.csv**: Precomputed image indices for each subtask in order to evaluate models on (all, summer2winter, winter2summer, day2night, night2day, old2new, new2old)

## ⚖ Legal

Copyright © Meta Platforms, Inc

This repository is MIT licensed.

[Terms of Use](https://opensource.facebook.com/legal/terms)

[Privacy Policy](https://opensource.facebook.com/legal/privacy)
