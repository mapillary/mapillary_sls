# Mapillary Street-level Sequences 

## Description

Mapillary Street-level Sequences (MSLS) is a large-scale long-term place recognition dataset that contains 1.6M street-level images. 

- ⬇️ Download: https://www.mapillary.com/dataset/places
- 📄 Paper: https://research.mapillary.com/publication/cvpr20c

## Package structure

- `images_vol_X.zip`: images, split into 6 parts for easier download
- `metadata.zip`: a single zip archive containing the metadata

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
	- utm
	- day/night
	- control_panel
	- qualitative_view_direction (forward, backward, sidewards)
	- unique_cluster

- **subtask_index.csv**: Precomputed image indices for each subtask in order to evaluate models on (all, summer2winter, winter2summer, day2night, night2day, old2new, new2old)

