# EC-523-final-project-Face-Mask-Prediction
------------

- Introduction 

During the pandemic, wearing masks makes it much more difficult for face recognition. In this way, it is necessary to improve the performance for face recognition of users. In this project, our task is generating a feature extraction backbone called ResSaNet [3]. By using ResSaNet, it would be able to integrate the self-attention module and CNN into the same network [3].  In this project, it would be difficult to capture discriminative features for face recognition. An additive Angular Margin Loss (ArcFace) would also be introduced to face this challenge [1].

------------
- Requirement 

  - Python
 
  - Pytorch 0.3.\*
  
  - CUDA 8.0 
  
------------
- Library 

Pandas (https://pandas.pydata.org/)

Numpy (https://numpy.org/)

PyTorch (https://pytorch.org/)

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
