# MDA-DUNet
MDA-DUNet: A Multiresolution Details Enhanced Attentive Dual-UNet for Hyperspectral and Multispectral Image Fusion

https://img.shields.io/badge/License-CC%20BY%204.0-green.svg](LICENSE)
https://img.shields.io/badge/PyTorch-1.7.0%2B-red.svg](https://pytorch.org/)

This repository is the official PyTorch implementation of the IEEE JSTARS 2023 paper: "A Multiresolution Details Enhanced Attentive Dual-UNet for Hyperspectral and Multispectral Image Fusion".

📖 Overview

Fusion-based super-resolution of Hyperspectral Images (HSIs) is a crucial technique to overcome the hardware limitations of HSI systems regarding spatial resolution. It aims to generate a High-Resolution HSI (HR-HSI) by fusing a Low-Resolution HSI (LR-HSI) with a High-Resolution Multispectral Image (HR-MSI). This paper proposes a Multiresolution Details Enhanced Attentive Dual-UNet (MDA-DUNet) for this task.

The entire network comprises two branches. The first branch is the Wavelet Detail Extraction Module, which applies the Discrete Wavelet Transform (DWT) to the HR-MSI to extract spatial detail features, processed through an encoder-decoder structure to capture multi-scale spatial information. The second branch is the Spatio-Spectral Fusion Module, designed to inject the extracted multi-scale detail features into the HSI for superior reconstruction. Furthermore, the network employs an Asymmetric Feature Selective Attention Module (AFSAM) to focus on important features at different scales. Extensive experiments on both simulated and real datasets demonstrate that the proposed MDA-DUNet achieves state-of-the-art performance qualitatively and quantitatively compared to several leading HSI super-resolution methods.

✨ Key Features

•   Innovative Dual-Branch Architecture: Integrates a Wavelet Detail Extraction Module and a Spatio-Spectral Fusion Module to fully exploit the multi-scale spatial information of the MSI and the spectral information of the HSI.

•   Efficient Feature Utilization: Explicitly extracts directional high-frequency details of the MSI via DWT and learns them at multiple scales through an encoder-decoder structure, maximizing the use of MSI's spatial details.

•   Advanced Attention Mechanism: Proposes an Asymmetric Feature Selective Attention Module (AFSAM) that adaptively selects and fuses asymmetric features from different scales using a spatial-spectral attention mechanism, leading to significant performance gains.

•   Superior Performance: Comprehensively validated on multiple public simulated datasets (CAVE, Harvard, ICVL, Chikusei) and real datasets (Hyperion-Sentinel-2, WV2). Experiments show that MDA-DUNet outperforms state-of-the-art methods across key metrics like MPSNR, SAM, ERGAS, UIQI, and MSSIM, while also demonstrating good noise robustness and high computational efficiency.

🎯 Network Architecture

The overall architecture of MDA-DUNet is illustrated in Figure 1 of the paper, consisting of four main parts:
1.  Wavelet Detail Extraction Module: Applies multi-level DWT to the input HR-MSI to extract horizontal, vertical, and diagonal high-frequency subbands. An encoder-decoder network then learns multi-scale detail features from them.
2.  Spatio-Spectral Encoding Module: Concatenates the upsampled LR-HSI with the low-frequency subband of the MSI. It progressively fuses multi-scale high-frequency features from the detail extraction module, generating a series of multi-scale spatio-spectral encoded features.
3.  Asymmetric Feature Selective Attention Module (AFSAM): Transforms and fuses features from different scales (with asymmetric spatial sizes and channel numbers) output by the encoding module. It employs spatial and channel attention mechanisms to generate weights for selectively enhancing important features.
4.  Spatio-Spectral Decoding Module: Aggregates the output of AFSAM, features from the detail extraction decoder, and skip-connection features from the encoder. It progressively upsamples the aggregated features to ultimately reconstruct the HR-HSI.

📊 Performance Summary (On CAVE Dataset, Scale Factor 8)

Method MPSNR (dB) ↑ SAM (degree) ↓ ERGAS ↓ UIQI ↑ MSSIM ↑

CNMF 34.30 7.89 2.60 0.771 0.939

HySure 34.78 11.55 2.42 0.804 0.911

CSU 35.69 7.88 2.21 0.799 0.953

DHSIS 46.30 3.85 0.66 0.924 0.990

DBIN 45.78 3.60 0.68 0.926 0.993

CNN-Fus 44.58 5.42 0.87 0.874 0.985

MoG-DCN 46.32 3.60 0.65 0.928 0.992

DUNet 46.28 4.33 0.67 0.913 0.989

MDA-DUNet (Ours) 47.19 3.46 0.60 0.934 0.993

Note: The table above shows partial average results on the CAVE dataset with a scale factor of 8, as reported in the paper. MDA-DUNet also achieves leading comprehensive performance with scale factors of 16, 32, and on other datasets (Harvard, ICVL, Chikusei). Please refer to the paper for more details.

🚀 Quick Start

1. Environment Setup

This code is implemented based on PyTorch. The experimental environment in the paper is described below. It is recommended to use a similar configuration.
•   OS: Windows 10 / Linux

•   Python: 3.7+

•   Deep Learning Framework: PyTorch >= 1.7.0 (Please install the PyTorch version that matches your CUDA version)

•   CPU: Intel(R) Core(TM) i7-9700

•   GPU: NVIDIA 2080Ti (or a GPU with higher performance)

•   Main Python Packages:
    numpy
    scipy
    matplotlib
    tqdm
    scikit-image
    
    (See requirements.txt for the complete dependency list)

2. Installation & Data Preparation

1.  Clone this repository
    git clone https://github.com/your_username/MDA-DUNet.git
    cd MDA-DUNet
    
2.  Install Python dependencies
    pip install -r requirements.txt
    
3.  Prepare Data
    ◦   Simulated Data: Download the http://www1.cs.columbia.edu/CAVE/databases/multispectral/, https://vision.seas.harvard.edu/hyperspec/, http://icvl.cs.bgu.ac.il/hyperspectral/, and http://naotoyokoya.com/Download.html datasets.

    ◦   Data Preprocessing: Simulate the degradation (spatial Gaussian blurring and downsampling, spectral response simulation) according to Section III-B of the paper. We provide a preprocessing script data/prepare_sim_data.py. Please modify the data paths and parameters according to the comments.

    ◦   Real Data: For Hyperion-Sentinel-2 and WV2 data, please refer to Section IV of the paper and the instructions in the code for processing.

3. Train the Model

Train the MDA-DUNet model on the CAVE dataset (taking scale factor 8 as an example):
python train.py --dataset CAVE --scale 8 --data_path ./data/CAVE --save_path ./checkpoints --epochs 1000 --lr 1e-4 --batch_size 32

Key Training Parameters (consistent with the paper):
•   Optimizer: Adam (β1=0.9, β2=0.999)

•   Learning Rate: 1e-4

•   Training Epochs: 1000

•   Batch Size: 32

•   Loss Function: L1 Loss

4. Test & Evaluation

Evaluate the trained model on the test set and compute metrics like MPSNR, SAM, ERGAS, UIQI, MSSIM:
python test.py --dataset CAVE --scale 8 --data_path ./data/CAVE/test --model_path ./checkpoints/best_model.pth


5. Predict on Your Own Data

We provide a simple prediction script for fusing your own LR-HSI and HR-MSI data pairs:
python predict.py --lrhsi_path ./your_data/lr.hdr --hrmsi_path ./your_data/msi.tif --model_path ./checkpoints/best_model.pth --output_path ./result.hdr


📁 Project Structure


MDA-DUNet/
├── data/                   # Data storage and preprocessing scripts
│   ├── prepare_sim_data.py
│   └── ...
├── src/                    # Source code directory
│   ├── models/
│   │   └── mda_dunet.py   # MDA-DUNet model definition (PyTorch)
│   ├── utils/
│   │   ├── dataset.py     # Data loading and preprocessing (PyTorch Dataset)
│   │   ├── wavelet.py     # Discrete Wavelet Transform (DWT) implementation
│   │   ├── attention.py   # AFSAM attention module
│   │   ├── metrics.py     # Evaluation metrics calculation (MPSNR, SAM, ERGAS, etc.)
│   │   └── ...
│   ├── train.py           # Main training script
│   ├── test.py            # Main testing and evaluation script
│   └── predict.py         # Single/Batch data prediction script
├── configs/               # Configuration files (optional, for different datasets/params)
├── checkpoints/           # Directory for saving trained models
├── results/               # Directory for test results (images, metric tables)
├── requirements.txt       # Project dependency list
└── README.md              # This file


📜 Citation

If you find our work helpful for your research, please cite our paper (Chinese reference format as in the original paper):

```bibtex
@article{fang2023multiresolution,
  title={A Multiresolution Details Enhanced Attentive Dual-UNet for Hyperspectral and Multispectral Image Fusion},
  author={Fang, Jian and Yang, Jingxiang and Khader, Abdolraheem and Xiao, Liang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={16},
  pages={638--655},
  year={2023},
  publisher={IEEE},
  doi={10.1109/JSTARS.2022.3228941}
}
```

📄 License

The code in this project is licensed under the MIT License. The underlying paper is licensed under a https://creativecommons.org/licenses/by/4.0/.

⁉️ Contact

If you have any questions, suggestions, or find a bug, please feel free to contact us or open an issue:
•   First Author: Jian Fang (fangjian@njust.edu.cn)

•   Maintainer: [Your GitHub Username]

Acknowledgments: This work was supported by the National Natural Science Foundation of China (No.61871226, No.61571230, No.62001226), the Key R&D Plan of Jiangsu Province (No.BE2018727), and the Natural Science Foundation of Jiangsu Province (No.BK20200465).
