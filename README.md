# Masked Face Recognition based on IResNet and ResSaNet
------------

- Introduction 

During the pandemic, wearing masks makes it much more difficult for face recognition. In this way, it is necessary to improve the performance of face recognition of users. In this project, our task is to generate a feature extraction backbone called Residual Networks, for example, IResNet and ResSaNet. By using Residual Networks, it would be able to integrate the self-attention module and CNN into the same network.  In this project, it would be difficult to capture discriminative features for face recognition. An additive Angular Margin Loss (ArcFace) would also be introduced to face this challenge.


------------
Installation Instructions 
- Requirement 

  - Python
 
  - Pytorch 0.3.\*
  
  - CUDA 8.0 
------------
- Library  

Numpy (https://numpy.org/)

PyTorch (https://pytorch.org/)

------------
- Dataset

The dataset we used is MS1M-RetinaFace, which containing over 10 million images of nearly 100,000 individuals. In addition, we add masks to the images in MS1M-RetinaFace by using the MaskTheFace toolkit. About 8% of total images are added masks successfully. The number of masked face images for each individual ranges from 0 to 180. To get a relatively large dataset, we keep images of the individual who has more than 100 masked face images. Besides, data augmentation is implemented by rotating and flipping the images. The cleaned dataset contains around 400 classes and ~60,000 images. 90% of the data is classified as training data and the rest is classified as testing data. 

- Dataset Processing (Adding Masks Installation)
- Steps to install MaskTheFace
We recommend to [make a new virtual environment](https://towardsdatascience.com/setting-up-python-platform-for-machine-learning-projects-cfd85682c54b) with Python 3 and install the dependencies. 
- Clone the repository
```
git clone https://github.com/aqeelanwar/MaskTheFace.git
```

- Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command

```
cd MaskTheFace
pip install –r requirements.txt
```

This will install the required packages in the activated Python environment.

- How to run MaskTheFace

```
cd MaskTheFace
- Generic
python mask_the_face.py --path <path-to-file-or-dir> --mask_type <type-of-mask> --verbose --write_original_image
```
- @misc{anwar2020masked,
title={Masked Face Recognition for Secure Authentication},
author={Aqeel Anwar and Arijit Raychowdhury},
year={2020},
eprint={2008.11104},
archivePrefix={arXiv},
primaryClass={cs.CV}

------------
-Module
  - ResNet
    
    It is a residual learning framework that can easily train substantially deeper networks.
    
    ![image](https://user-images.githubusercontent.com/90427304/162338241-b4296885-482d-40a5-a407-bcf2981255be.png)

  - IResNet

    IResNet was inspired by the structure of ResNet. 
    
    ![image](https://user-images.githubusercontent.com/90427304/162337963-e6ba3262-16b1-4fe3-b2b6-5839b8377596.png)

  - ResSaNet

    ResSaNet was inspired by the structure of IResNet.
    
    ![image](https://user-images.githubusercontent.com/90427304/162338283-564a7f66-0e18-49ba-847b-ebd394cca820.png)

------------
- Loss

ArcFace loss is a loss function used in face recognition tasks. It is the proposed based on softmax loss.

![image](https://user-images.githubusercontent.com/90427304/162339549-b53a9e00-39ce-4297-bd34-1b863eb4f4a0.png)


------------
- Maintainers

 Fei Gao           gaof@bu.edu
 
 Shiwen Tang       shiwent@bu.edu
 
 Shiyu Hu          shiyuhu@bu.edu
 
------------
- REFERRENCE

[1] J. Deng, J. Guo, N. Xue, S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699), 2019.

[2] J. Deng, J. Guo, X. An, Z. Zhu, S. Zafeiriou. Masked face recognition challenge: The insightface track report. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1437-1444), 2021

[3] W. Y. Chang, M. Y. Tsai, and S. C. Lo. ResSaNet: A Hybrid Backbone of Residual Block and Self-Attention Module for Masked Face Recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1468-1476), 2021.

[4] Y. Guo, L. Zhang, Y. Hu, X. He, J. Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. In European conference on computer vision (pp. 87-102). Springer, Cham, 2016.

[5] A. Anwar, A. Raychowdhury. Masked Face Recognition for Secure Authentication, arXiv.org, 2020. 

[6] I. C. Duta, L. Liu, F. Zhu, L. Shao.  Improved residual networks for image and video recognition. In 2020 25th International Conference on Pattern 
