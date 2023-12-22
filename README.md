# BAE-Net: A Low complexity and high fidelity Bandwidth-Adaptive neural network for speech super-resolution
This is the repo of the manuscript "BAE-Net: A Low complexity and high fidelity Bandwidth-Adaptive neural network for speech super-resolution", which is accepted to ICASSP 2024 (https://arxiv.org/abs/2312.13722). Some audio samples are provided here and the code for network is released soon.

 Speech bandwidth extension (BWE) has demonstrated promising performance in enhancing the perceptual speech quality in real communication systems. Most existing BWE researches primarily focus on fixed upsampling ratios, disregarding the fact that the effective bandwidth of captured audio may fluctuate frequently due to various capturing devices and transmission conditions. In this paper, we propose a novel streaming adaptive bandwidth extension solution dubbed BAE-Net, which is suitable to handle the low-resolution speech with unknown and varying effective bandwidth. To address the challenges of recovering both the high-frequency magnitude and phase speech content blindly, we devise a dual-stream architecture that incorporates the magnitude inpainting and phase refinement. For potential applications on edge devices, this paper also introduces BAE-NET-lite, which is a lightweight, streaming and efficient framework. Quantitative results demonstrate the superiority of BAE-Net in terms of both performance and computational efficiency when compared with existing state-of-the-art BWE methods.

 
### System flowchart of BAE-Net
![image](https://github.com/yuguochencuc/BAE-Net/assets/51236251/3738cc0f-46e8-4833-9f0b-294e6654fd85)



### Results:
![image](https://github.com/yuguochencuc/BAE-Net/assets/51236251/43a9fa0a-0af2-406a-9f41-dd464fff3d44)

![image](https://github.com/yuguochencuc/BAE-Net/assets/51236251/af0b86fc-b6d4-44b8-ae09-41922d9f1eb7)


### Visualization of spectrograms of captured LR speech, generated HR speech by AERO and BAE-Net, and the HR reference.

![image](https://github.com/yuguochencuc/BAE-Net/assets/51236251/d2415f51-606a-45bb-8561-e89a71c2d5d5)


