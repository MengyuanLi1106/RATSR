# An Intelligent UAV Thermal Imaging System with Arbitrary-Scale Super-Resolution 

### MengyuanLi, Changhong Fu*, Haobo Zuo, Guangze Zheng, Ziyu Lu, Jia Pan
\* Corresponding author.

## Abstract
Typical unmanned aerial vehicle (UAV) thermal imaging systems are often limited by the low spatial resolution of onboard thermal sensors, which restricts the ability to provide clear and flexible-resolution perception during real-world missions. This limitation becomes more critical when multi-scale observation and zoom are required, while existing super-resolution (SR) methods are difficult to deploy due to their high computational cost. To address this challenge, we propose an intelligent UAV thermal imaging system with arbitrary-scale thermal SR. Specifically, a novel lightweight ring convolution (LRingConv) is introduced to enhance texture-aware feature extraction by emphasizing boundary and high-frequency thermal structures. Based on LRingConv, a lightweight ring convolution module (LRCM) is designed to effectively combine texture-sensitive and structural feature representations. In addition, a novel anisotropic diffusion field upsampler is incorporated to explicitly model anisotropic spatial relationships in thermal images, enabling direction-adaptive feature aggregation for continuous-scale reconstruction while maintaining low computational overhead. Extensive experiments on the public thermal SR benchmark demonstrate that the proposed system achieves superior reconstruction performance. Furthermore, real-world application results show that the system achieves real-time performance under different operational modes, delivering 79.59 FPS for part-frame zoom-in SR and 20.03 FPS for $\times$2 full-frame global-view SR on the UAV thermal imaging system.

## Dataset
### UAV-TSR++ Dataset
📦 **Download (Baidu Netdisk)**  
**Link:** https://pan.baidu.com/s/1Q3D_CncX4D21twnvbhzmMQ  
**Access Code:** `atsr`

## About Code
### 1. Environment setup
This code has been tested on Ubuntu 22.04, Python 3.10.15, Pytorch 1.13.0, CUDA 11.7.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test

For the visual demo, put a few ground-truth thermal images into:

```text
RATSR/data/demo/GT/
```

For example:

```text
RATSR/
|-- data/
|   |-- demo/
|   |   |-- GT/
|   |       |-- xxxx.png
|   |       |-- ...
```

To run the visual demo, use:

```bash 
python demo.py                                
```

This script reads images from `data/demo/GT`, then performs super-resolution using the provided RATSR model. The super-resolved outputs are saved in the `output` directory.

For quantitative evaluation, put the downloaded test-set ground-truth thermal images into:

```text
RATSR/data/test/GT/
```

For example:

```text
RATSR/
|-- data/
|   |-- test/
|       |-- GT/
|           |-- xxxx.png
|           |-- ...
```

The default test configuration is `configs/test.yaml`, where the test image path is set as:

```yaml
root_path: data/test/GT
```

If the dataset is placed in another folder, please modify `root_path` in `configs/test.yaml` accordingly.

To evaluate PSNR and SSIM on the test set, use:

```bash
python test.py --scale 2 3 4
```

By default, `test.py` loads:

```text
configs/test.yaml
checkpoints/best.pth
```

### 3. Contact
If you have any questions, please contact me.

Mengyuan Li

Email: [mengyuanli@tongji.edu.cn](mengyuanli@tongji.edu.cn)


## References 

```

```
