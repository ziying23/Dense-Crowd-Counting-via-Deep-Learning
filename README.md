# Dense-Crowd-Counting-via-Deep-Learning
A Final Year Project in PyTorch version repo by implementing CNN to conduct dense estimation. This project is done by author Leong Zi Ying from The National University of Malaysia. 

Crowd analysis is an important research topic to understand the behaviours and dynamics of individuals in crowded scenes. Crowd analysis is essential for crowd management especially in public events, for example public space design or virtual environments. With the increase of population, there is limitation of space for mass gathering which may further lead to congestion if not being managed well. Hence, the crowd density parameter in the surveillance system need to be managed well. However, previous work showed that it is difficulties to estimate the number of people in the crowd due to scale variations of individuals in a crowd as a results of perspective distortion. Therefore, this project aims to develop an algorithm that can extract scale-relevant features to adapt scale variation. The proposed model can cope with dense crowd for crowd counting. To overcome the scale variation problem, this project will propose a novel framework based on convolutional neural network for crowd counting. This proposed model implemented VGG-16 as the backbone by aggregating with attention network and pyramid dilated convolution which uses dilated kernels to enlarge reception field and to replace pooling operations. Therefore, the model is named as Attention Pyramid Dilated Netwok (APDNet). The output feature density map with crowd count will be used for evaluation of the model. Extensive experiments will be conducted using public benchmark datasets such as ShanghaiTech Part A and Part B.

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

## Citation

```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}
```

```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```
