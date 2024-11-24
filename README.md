#VCA    
Virtual coil augmentation for MR coil extrapoltion via deep learning    
C. Yang, X. Liao, L. Zhang, M. Zhang, Q. Liu    
Magnetic Resonance Imaging 95, 1-11, 2023   
https://www.sciencedirect.com/science/article/pii/S0730725X22001722   

Magnetic resonance imaging (MRI) is a widely used medical imaging modality. However, due to the limitations in hardware, scan time, and throughput, it is often clinically challenging to obtain high-quality MR images. In this article, we propose a method of using artificial intelligence to expand the coils to achieve the goal of generating the virtual coils. The main characteristic of our work is utilizing dummy variable technology to expand/extrapolate the receive coils in both image and k-space domains. The high-dimensional information formed by coil expansion is used as the prior information to improve the reconstruction performance of parallel imaging. Two main components are incorporated into the network design, namely variable augmentation technology and sum of squares (SOS) objective function. Variable augmentation provides the network with more high-dimensional prior information, which is helpful for the network to extract the deep feature information of the data. The SOS objective function is employed to solve the deficiency of k-space data training while speeding up convergence. Experimental results demonstrated its great potentials in accelerating parallel imaging reconstruction.       

## The training pipeline of VCA
 <div align="center"><img src="https://github.com/yqx7150/VCA/blob/main/Fig1-VCA.png"> </div>
 
Top line: VCA-I that conducts the coil expansion in image domain; Bottom line: VCA-K that conducts the coil expansion in k-space domain. Among them, variable augmentation technology is used in the stage of image pre-processing. The loss function is composed of a forward loss function and a reverse loss function.

##  The pipeline of VCA
 <div align="center"><img src="https://github.com/yqx7150/VCA/blob/main/Fig2-VCA.png"> </div>

Invertible model is composed of both forward and inverse processes.

##  Reconstruction results of VCA-I on full-sampling Brain dataset
<div align="center"><img src="https://github.com/yqx7150/VCA/blob/main/Fig3-VCA.png"> </div>

Top: (a) Reference image of Brain dataset; Reconstruction results of (b) 2ch→12ch; (c) 4ch→12ch; (d) 6ch→12ch; Bottom: The 10× residuals between the reference images and reconstruction images.

## Reconstruction results of VCA-I on full-sampling Cardiac dataset
<div align="center"><img src="https://github.com/yqx7150/VCA/blob/main/Fig4-VCA.png"> </div>

 Top: (a) Reference image of Cardiac dataset; Reconstruction results of (b) 2ch→20ch; (c) 6ch→20ch; (d) 10ch→20ch; Bottom: The 3× residuals between the reference images and reconstruction images.
      
### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/algorithm-overview.png" width = "800" height = "500"> </div>
 Some examples of invertible and variable augmented network: IVNAC, VAN-ICC, iVAN and DTS-INN.

  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)    

  * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)               

  * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

  * Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN)        

  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       
    
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
