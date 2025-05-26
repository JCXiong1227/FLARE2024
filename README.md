This repository is the official implementation of our participation in [FLARE competition](https://www.codabench.org/competitions/2320/). Experiments on the MICCAI FLARE 2024 TASK2 challenge leaderboard validate promising performance achieving high segmentation accuracy with average Dice similarity coefficients and  Normalized Surface Dice (NSD) of 90.02 % and 95.51 % for multi-organs segmentation respectively. In this repository, we release the inference process that only including phase_2



## Methods

The detailed methodology can be found in the network architecture design of the second stage in the main branch.
  

- Evaluation metrics
  - Dice Similarity Coefficient (DSC) and Normalized Surface Dice (NSD)
  



## Environments and Requirements

- Ubuntu 20.04/Ubuntu 22.04
- CPU:Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz, RAM: 1.0 Ti; 3200 MT/S, GPU: NVIDIA A800 80G
- CUDA 11.6
- python 3.8.15
- monai 1.2.0

To install requirements:

```setup
pip install -r requirements.txt
```

## ValidResults
The validation set consists of two subsets: Testing and Tuning. For detailed information about the data sources, please refer to the official webpage of FLARE2024 Task 2. The inference results for each subset are provided separately in the ValidResults folder.

## How to use this code
### Inference On CPU
Inferencing files can be found in  folder "flare2024OneStageinference". To execute inference on a GPU, please modify certain sections of the code in ./flare2024OneStageinference/infer.py. Run in terminal: 
```
cd flare2024OneStageinference
python infer.py
```

## Docker
To provide detailed information, we have made the Docker image publicly available at: 

```twostage
pip install -r requirements.txt
```
```onestage
pip install -r requirements.txt
```
