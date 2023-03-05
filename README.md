# MBSNet
MBSNet is implemented in Pytorch, a new medical image segmentation method based on multi-branch segmentation network with synchronous learning of local and global information.
# Abstract
In recent years, there have been several solutions to medical image segmentation, such as U-shaped structure, transformer-based network, and multi-scale feature learning method. However, their network parameters and real-time performance are
often neglected and cannot segment boundary regions well. The main reason is that such networks have deep encoders,
a large number of channels, and excessive attention to local information rather than global information, which is crucial to
the accuracy of image segmentation. Therefore, we propose a novel multi-branch medical image segmentation network
architecture MBSNet. We first design two branches using a parallel residual mixer(PRM) module and dilate convolution block
to capture the local and global information of the image. At the same time, a SE-Block and a new spatial attention module
enhance the output features. Considering the different output features of the two branches, we adopt a cross-fusion method
to effectively combine and complement the features between different layers. MBSNet was tested on three datasets ISIC2018,
Kvasir, and BUSI. The combined results show that MBSNet is lighter, faster, and more accurate. Specifically, for a 320×320
input, MBSNet’s FLOPs is 10.68G, with an F1-Score of 84.52% on the Kvasir test dataset, well above 76.5% for UNet++ with
FLOPs of 216.55G.
![image](https://user-images.githubusercontent.com/38003218/222957253-63cfdb2b-d9c3-4226-848c-32c21745d4ac.png)
# Using the code:
The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:

``` Python
git clone https://github.com/jeya-maria-jose/UNeXt-pytorch
cd MBSNet
```
