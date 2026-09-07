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

