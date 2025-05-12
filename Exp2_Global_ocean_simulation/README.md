


## Installation

- cuda 11.8

```
conda env create -f environment.yml
conda activate triton
```

## Notes

The full project is avilable on [Hugging Face](https://huggingface.co/easylearning/Triton_Earth_V1/tree/main), you can find the pretrained model, test data on Hugging Face and put them in the same location.

## Data Structure

Preparing the train, valid, and test data as follows:

```
./data/
|--train
|  |--1993.h5
|  |--1994.h5
|  |--......
|  |--2016.h5
|  |--2017.h5
|--valid
|  |--2018.h5
|  |--2019.h5
|--test
|  |--2020.h5
|--mean_s_t_ssh.npy
|--std_s_t_ssh.npy
|--climate_mean_s_t_ssh.npy
|--land_mask.h5
```

## Inference

```
sh inference.sh
```
   
## Training

- **Single-node Multi-GPU Training**

```
sh train_single_node_and_multi_gpus.sh
```

- **Multi-node Multi-GPU Training**

```
sh train_multi_nodes_and_multi_gpus.sh
```

