# Dense-Crowd-Counting-via-Deep-Learning
A Final Year Project in PyTorch version repo by implementing CNN to conduct dense estimation.

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.0

PyTorch: 1.0.1

CUDA: 9.2

## Ground Truth

Please follow the `main.ipynb` to generate the ground truth. It shall take some time to generate the dynamic ground truth. Note you need to generate your own json file.

## Training Process

Try `python train.py train.json val.json 0 0` to start training process.

## Validation

Follow the `val.ipynb` to try the validation. You can try to modify the notebook and see the output of each image.
