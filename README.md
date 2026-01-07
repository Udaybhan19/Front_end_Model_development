# 🌱 Plant Disease Detection & Segmentation Mobile App

This repository provides an **AI-powered plant disease detection and semantic segmentation system** designed for **mobile applications**. The segmentation pipeline is built on top of **MMSegmentation** using the **InternImage** backbone, with a **FastAPI-based backend** for real-time inference.

The system focuses on **20 common plant diseases** (e.g., spot, blight, rot, rust) and is optimized for accuracy, scalability, and deployment readiness.

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

### 2️⃣ Create and Activate Conda Environment
```bash
conda create -n plantdisease python=3.9 -y
conda activate plantdisease

### 3️⃣ Install CUDA & PyTorch
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y

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

## Compile CUDA Operators (DCNv3)

### Before compiling, please use the nvcc -V command to check whether your nvcc version matches the CUDA version of PyTorch.
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py


## 📁 Data Preparation

Two Python scripts are provided for dataset preparation.

### `prepare_dataset.py`

**Description**  
Converts raw plant disease datasets into an MMSegmentation-compatible format and creates train/test splits.

```bash
python prepare_dataset.py --data_root /path/to/raw_dataset

## 🧠 Models & Weights

### Download Pretrained Weights

Download the pretrained model weights from the provided **Google Drive** link.

Create a directory to store the model weights:

```bash
mkdir model_weights

lace the downloaded file inside the directory:

model_weights/
└── internimage_segmentation.pth

# 🚀 API Setup (FastAPI Backend)
## API Files Overview
### The API module consists of the following files:

### disease_config.py
### Contains plant disease metadata such as disease names, categories, and descriptions.
### model_config.py
### Defines model weight paths and MMSegmentation configuration file locations.
### api_server.py
### Implements the FastAPI server for inference and mobile application integration.

## Generate API Key
###Generate an API key using the following command:
```bash
python generate_api_key.py


## Start API Server
### Launch the FastAPI server:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000


