# CECT
<a href='https://arxiv.org/abs/2302.02314'><img src='https://img.shields.io/badge/ArXiv-2302.02314-red' /></a> 

Official Pytorch Implementation for Paper “CECT: Controllable Ensemble CNN and Transformer for COVID-19 Image Classification”

To train CECT on your own dataset, run the below command in the terminal:
```
python train.py
```

You should have a "recording" folder located at the root.

The organization of the dataset should follow:
```
├── datasets
    ├── your_dataset_name
        ├── training
        |   ├── class_1
        |   |   ├── img_1.jpg
        |   │   ├── img_2.jpg
        |   │   ├── ...
        |   ├── class_2
        |   |   ├── img_a.jpg
        |   │   ├── img_b.jpg
        |   │   ├── ...
        |   ├── ...
        └── validation
        └── test
```

##
If you find CECT useful for your research, please cite our paper as:
```
@article{liu2024cect,
  title={CECT: Controllable ensemble CNN and transformer for COVID-19 image classification},
  author={Liu, Zhaoshan and Shen, Lei},
  journal={Computers in Biology and Medicine},
  pages={108388},
  year={2024},
  publisher={Elsevier}
}
```
