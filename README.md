# Masked Face Recognition based on IResNet and ResSaNet

## Introduction 

Wearing masks during a pandemic makes face identification much more difficult. In this case, it is vital to improve user face recognition performance. Our aim in this project is to generate a feature extraction backbone called Residual Networks, for example, IResNet and ResSaNet. By using Residual Networks, it would be able to integrate the self-attention module and CNN into the same network.  In this project, it would be difficult to capture discriminative features for face recognition. An additive Angular Margin Loss (ArcFace) would also be introduced to face this challenge.


## Requirement 

  - Python 3.6  
  - Pytorch 0.3.\*
  - CUDA 8.0 

## Library  
 
  - Numpy (https://numpy.org/)
  - PyTorch (https://pytorch.org/)


## Dataset

The dataset we used is [MS1M-RetinaFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_), which containing over 10 million images of nearly 100,000 individuals. In addition, we add masks to the images in MS1M-RetinaFace by using the MaskTheFace toolkit. About 8% of total images are added masks successfully. The number of masked face images for each individual ranges from 0 to 180. To get a relatively large dataset, we keep images of the individual who has more than 100 masked face images. Besides, data augmentation is implemented by rotating and flipping the images. The cleaned dataset contains around 400 classes and ~60,000 images. 90% of the data is classified as training data and the rest is classified as testing data. 

### Dataset Processing (Adding Masks)
We use the [MaskTheFace toolkit](https://github.com/aqeelanwar/MaskTheFace) created by Aqeel Anwar and Arijit Raychowdhury. <br />

## Models
  - [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
    
    It is a residual learning framework that can easily train substantially deeper networks.
    
    ![image](https://user-images.githubusercontent.com/90427304/162338241-b4296885-482d-40a5-a407-bcf2981255be.png)
    
    Reference of ResNet: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

  - IResNet [6]

    IResNet was inspired by the structure of ResNet.
    
    ![image](https://user-images.githubusercontent.com/90427304/162337963-e6ba3262-16b1-4fe3-b2b6-5839b8377596.png)
    
    Reference of IBasic block and IResNet: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

  - ResSaNet [3]

    ResSaNet was inspired by the structure of IResNet. 
    
    ![image](https://user-images.githubusercontent.com/90427304/162338283-564a7f66-0e18-49ba-847b-ebd394cca820.png)
    
    <img src="https://github.com/shiyuhu1933/ResSaNet/blob/main/readme/Screen%20Shot%202022-05-06%20at%203.53.16%20AM.png" alt="drawing" width="600"/>
    
    <img src="https://github.com/shiyuhu1933/ResSaNet/blob/main/readme/Screen%20Shot%202022-05-06%20at%203.54.27%20AM.png" alt="drawing" width="1000"/>
    
    Reference of MHSA block: https://github.com/leaderj1001/BottleneckTransformers
    
    ### How to change models?
    - ResNet-50
    ```
    model = resnet50().cuda()
    ```
    - iResNet-50
    ```
    model = iResNet(BasicBlock, [3,4,14,3]).cuda()
    ```
    - iResNet-100
    ```
    model = iResNet(BasicBlock, [3,13,30,3]).cuda()
    ```
    - ResSaNet-50
    ```
    model = ResSaNet([iBasic, SE_iBasic_F, SE_iBasic_F, IBT], [3,4,14,3]).cuda()
    ```
    - ResSaNet-100
    ```
    model = ResSaNet([iBasic, SE_iBasic_F, SE_iBasic_F, IBT], [3,13,30,3]).cuda()
    ```

## Loss

ArcFace loss [1] is a loss function used in face recognition tasks. It is the proposed based on softmax loss.

![image](https://user-images.githubusercontent.com/90427304/162339549-b53a9e00-39ce-4297-bd34-1b863eb4f4a0.png)


## Results

For all the models, we set the learning rate to 0.05, the optimizer to SGD, and the weight decay to 5e-4.By training the different Residual models with the same number of batches 512 on the Masked-Face Datasets, we could get the accuracy as follows:

| Model         | Accuracy      |
| ------------- | ------------- |
| ResNet        | 89.62%        | 
| IResNet-50    | 89.89%        |
| IResNet-100   | 89.07%        |
| ResSaNet-50   | 90.17%        |
| ResSaNet-100  | 92.08%        |


By training the baseline model IResNet-50 on two datasets, we got the accuracy as follows:

| Datasets       | Accuracy      |
| -------------  | ------------- |
| Masked-Face    | 89.90%        | 
| Non-Masked-Face| 94.94%        |



## Maintainers

 Fei Gao           gaof@bu.edu
 
 Shiwen Tang       shiwent@bu.edu
 
 Shiyu Hu          shiyuhu@bu.edu

## REFERRENCE

[1] J. Deng, J. Guo, N. Xue, S. Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690-4699), 2019.

[2] J. Deng, J. Guo, X. An, Z. Zhu, S. Zafeiriou. Masked face recognition challenge: The insightface track report. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1437-1444), 2021

[3] W. Y. Chang, M. Y. Tsai, and S. C. Lo. ResSaNet: A Hybrid Backbone of Residual Block and Self-Attention Module for Masked Face Recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1468-1476), 2021.

[4] Y. Guo, L. Zhang, Y. Hu, X. He, J. Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. In European conference on computer vision (pp. 87-102). Springer, Cham, 2016.

[5] A. Anwar, A. Raychowdhury. Masked Face Recognition for Secure Authentication, arXiv.org, 2020. 

[6] I. C. Duta, L. Liu, F. Zhu, L. Shao.  Improved residual networks for image and video recognition. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 9415-9422), 2021.

[7] A, Srinivas., T, Lin., N, Parmar., J, Shlens., P, Abbeel., A, Vaswani. (2021). Bottleneck Transformers for Visual Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16514-16524), 2021.



