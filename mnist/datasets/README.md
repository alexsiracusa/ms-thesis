# Datasets
MNIST-like datasets were gathered from the following sources and standardized to black and white 50x50 images if not already in that format.  Datasets that did not already have an existing train/test split were randomly split 80/20.  Datasets with an existing validation set where merged into the test set (this only applies to MedMNIST datasets).

> **_NOTE:_** Sign MNIST skips label=9 as they don't have the letter J.  Thus there are actually 24 classes but to allow cross-entropy loss to work without changing the labels I added an additional 25th class.

> **_NOTE:_**  EMNIST Letters labels start at 1 instead of 0, so I used the same solution as above.

| Name            | Num Classes | Num Train | Num Test  | URL                                                                             |
|-----------------|-------------|-----------|-----------|---------------------------------------------------------------------------------|
| MNIST           | 10          | 60,000    | 10,000    |                                                                                 |
| EMNIST Letters  | 27*         | 88,800    | 14,800    | https://github.com/hosford42/EMNIST                                             |
| EMNIST Balanced | 47          | 112,800   | 18,800    | https://github.com/hosford42/EMNIST                                             |
| FashionMNIST    | 10          | 60,000    | 10,000    | https://github.com/zalandoresearch/fashion-mnist                                |
| KMNIST          | 10          | 60,000    | 10,000    | https://github.com/rois-codh/kmnist                                             |
| CIFAR10         | 10          | 50,000    | 10,000    | https://www.cs.toronto.edu/~kriz/cifar.html                                     |
| Sign MNIST      | 25*         | 27,455    | 7,172     | https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download |
| Chinese MNIST   | 15          | 15,000    | -         | https://data.ncl.ac.uk/articles/dataset/Handwritten_Chinese_Numbers/10280831/1  |
| Kannada MNIST   | 10          | 60,000    | 10,000    | https://www.kaggle.com/datasets/higgstachyon/kannada-mnist                      |
| Dig MNIST       | 10          | 10,240    | -         | https://www.kaggle.com/datasets/higgstachyon/kannada-mnist                      |
| Overhead MNIST  | 10          | 8,519     | 1,066     | https://www.kaggle.com/datasets/datamunge/overheadmnist/                        |
| Simpsons MNIST  | 10          | 8,000     | 2,000     | https://github.com/alvarobartt/simpsons-mnist                                   |
| notMNIST        | 10          | 18,724    | -         | https://www.kaggle.com/datasets/lubaroli/notmnist                               |
| PathMNIST       | 9           | 89,996    | 17,184    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| ChestMNIST      | 14 or 2     | 78,468    | 33,652    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| DermaMNIST      | 7           | 7,007     | 3,008     | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| OCTMNIST        | 4           | 97,477    | 11.832    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| PneumoniaMNIST  | 2           | 4,708     | 1,148     | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| RetinaMNIST     | 5           | 1,080     | 520       | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| BreastMNIST     | 2           | 546       | 234       | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| BloodMNIST      | 8           | 11,959    | 5,133     | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| TissueMNIST     | 8           | 165,466   | 70,920    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| OrganAMNIST     | 11          | 34,561    | 24,269    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| OrganCMNIST     | 11          | 12,975    | 10,608    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |
| OrganSMNIST     | 11          | 13,932    | 11,279    | https://github.com/MedMNIST/MedMNIST?tab=readme-ov-file                         |



## Other one's I'm not using:
Skin Cancer MNIST - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data
- looks annoying to download

HASY - https://github.com/MartinThoma/HASY
- too many (300+) classes

Afro MNIST - https://github.com/Daniel-Wu/AfroMNIST
- don't like synthetic data

## List of MNIST-like datasets
https://www.simonwenkel.com/lists/datasets/list-of-mnist-like-datasets.html
