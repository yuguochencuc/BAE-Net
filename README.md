### Fork from BAE-Net https://github.com/yuguochencuc/BAE-Net 
Since there seem to be some issues in the source code, I have fixed few bugs here.

#### Issue 1 – ERB module not compressing when erb_dim = input_size
In the larger BAE‑Net configuration, erb_dim is set equal to input_size. As a result, the ERB module performs no dimensionality reduction. However, the code still executes a matrix multiplication followed by a square root operation. 

#### Issue 2 – Incorrect self.inter_com index in the Phase Encoder branch
In the Phase Encoder branch, the code contains the following line:

inter_4 = self.inter_com[2](enc_5, mag_list[4])
This incorrectly reuses self.inter_com[2] instead of self.inter_com[3]. As a result, self.inter_com[3] is never involved in either the forward pass or backpropagation, which may lead to unintended model behavior or performance degradation.

#### Issue 3 - Discriminator...
