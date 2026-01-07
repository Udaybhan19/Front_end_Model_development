# 🌱 Plant Disease Detection & Segmentation Mobile App
 
This repository provides an **AI-powered plant disease detection and semantic segmentation system** designed for **mobile applications**. The segmentation pipeline is built on top of MSegmentation using the SOTA Segmentation model as backbone, with a **FastAPI-based backend** for real-time inference.
 
The system focuses on **20 common plant diseases** (e.g., spot, blight, rot, rust) for unknown plant species. In addition, it targets 12 major crops, each containing crop-specific diseases, including apple, grape, banana, rice, wheat, soybean, tomato, potato, bean, cucumber, cabbage, and corn, to ensure scalability and deployment readiness for real-world applications.
 
---
 
## ✨ Features
 
- Semantic segmentation using **InternImage**
- Built on **MMSegmentation**
- FastAPI backend for mobile integration
- GPU-accelerated inference
 
---
 
## 📦 Installation
 
### 1️⃣ Clone the Repository
 
```bash
git clone https://github.com/Rashmi/Plant_Disease.git
 ```
### 2️⃣ Create and Activate Conda Environment
```bash
conda create -n plantdisease python=3.9 -y
conda activate plantdisease
 ```
### 3️⃣ Install CUDA & PyTorch
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
 ```
### 4️⃣ Install Core Dependencies
```bash
pip install opencv-python
pip install -U openmim
mim install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
pip install yapf==0.40.1
pip install termcolor yacs pyyaml scipy
pip install numpy==1.26.4   # numpy < 2.0 required
pip install pydantic==1.10.13
```
 ---
## Compile CUDA Operators (DCNv3)
 
### Before compiling, please use the nvcc -V command to check whether your nvcc version matches the CUDA version of PyTorch.
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
 ```
  ---
## 📁 Data Preparation
 
Two Python scripts are provided for dataset preparation used to updates the MMSegmentation dataset directory created during installation in the plantdisease conda environment and generates crop-specific, MMSegmentation-compatible dataset files. 
 
```bash
python Init_All_Creation.py
python Plant_Disease_All_Creation.py 
 ```
## 🧠 Models & Weights
 
### Download Pretrained Weights
 
Download the pretrained model weights from the provided **Google Drive** link.
 
Create a directory to store the model weights:
 
```bash
mkdir model_weights
 ```
Place the downloaded file inside the directory:
```bash 
model_weights/
└── Unknown_Disease.pth
└── Apple_Disease.pth
└── Banana_Disease.pth
...
  ```
---
## 🚀 API Setup (FastAPI Backend)

### API Files Overview

The API module consists of the following files:

- **`API_Disease_Config.py`**  
  Stores plant disease metadata, including disease names, crop_overview, and prevention.

- **`API_Model_Config.py`**  
  Defines the model weight and configuration (config.) file paths for each crop type used during model loading, and specifies the disease class names associated with each crop.

- **`API_Main.py`**  
  Implements the FastAPI application to handle image uploads, perform inference, return results for mobile app integration, and generate API keys.

---

### Generate API Key

Generate an API key by running the following command:

```bash
python API_Main.py
```
