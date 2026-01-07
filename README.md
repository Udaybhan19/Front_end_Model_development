# 🌱 Plant Disease Detection & Segmentation Mobile App

This repository provides an **AI-powered plant disease detection and semantic segmentation system** designed for **mobile applications**. The segmentation pipeline is built on top of **MMSegmentation v0.27.0** using the **InternImage** backbone, with a **FastAPI-based backend** for real-time inference.

The system focuses on **20 common plant diseases** (e.g., spot, blight, rot, rust) and is optimized for accuracy, scalability, and deployment readiness.

---

## ✨ Key Features

- Semantic segmentation using **InternImage backbone**
- Built on **MMSegmentation v0.27.0**
- Mobile-friendly **FastAPI inference APIs**
- Modular dataset loader
- Easy model weight integration
- Ready for **GPU acceleration**
- Research- and production-ready codebase

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage
2️⃣ Create and Activate Conda Environment
bash
Copy code
conda create -n internimage python=3.9 -y
conda activate internimage
3️⃣ Install CUDA & PyTorch
Requirements

CUDA ≥ 10.2

cuDNN ≥ 7

Example: Install PyTorch 1.11 with CUDA 11.3

bash
Copy code
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
-f https://download.pytorch.org/whl/torch_stable.html
4️⃣ Install Core Dependencies
⚠️ Do not install OpenCV via conda, as it breaks GPU support in torchvision.

bash
Copy code
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
5️⃣ Install MMSegmentation & Related Libraries
bash
Copy code
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
6️⃣ Install Remaining Requirements
bash
Copy code
pip install termcolor yacs pyyaml scipy
pip install numpy==1.26.4   # numpy < 2.0 required
pip install pydantic==1.10.13
⚙️ Compile CUDA Operators (DCNv3)
Before compiling, verify CUDA compatibility:

bash
Copy code
nvcc -V
Compile DCNv3 operators:

bash
Copy code
cd ops_dcnv3
sh make.sh
Run unit tests:

bash
Copy code
python test.py
✔️ All checks should return True.

📁 Data Preparation
Two Python scripts are provided for dataset preparation.

prepare_dataset.py
Purpose

Converts raw datasets into MMSegmentation-compatible format

Organizes images and masks into train/test splits

bash
Copy code
python prepare_dataset.py --data_root /path/to/raw_dataset
validate_dataset.py
Purpose

Validates image–mask alignment

Checks label consistency and missing annotations

bash
Copy code
python validate_dataset.py --data_root /path/to/processed_dataset
🧠 Models & Weights
Download Pretrained Weights
Download pretrained model weights from the provided Google Drive link

Create a directory for model weights:

bash
Copy code
mkdir model_weights
Place the downloaded model file inside:

text
Copy code
model_weights/
└── internimage_segmentation.pth
🚀 API Setup (FastAPI Backend)
API Files Overview
The API module contains three key files:

disease_config.py
Contains plant disease metadata (names, categories, descriptions)

model_config.py
Stores model paths, config file locations, and weight mappings

api_server.py
FastAPI application for inference and mobile integration

Generate API Key
bash
Copy code
python generate_api_key.py
Start the API Server
bash
Copy code
uvicorn api_server:app --host 0.0.0.0 --port 8000
🖼️ Image Demo
bash
Copy code
python demo/image_demo.py \
configs/plant_disease/internimage_config.py \
model_weights/internimage_segmentation.pth \
--img demo/sample.jpg
📊 Evaluation
bash
Copy code
python tools/test.py \
configs/plant_disease/internimage_config.py \
model_weights/internimage_segmentation.pth \
--eval mIoU
🏋️ Training
bash
Copy code
python tools/train.py \
configs/plant_disease/internimage_config.py
🖥️ Manage Jobs with Slurm
bash
Copy code
bash tools/slurm_train.sh \
internimage_job \
configs/plant_disease/internimage_config.py \
8
📦 Export Model
bash
Copy code
python tools/export.py \
configs/plant_disease/internimage_config.py \
model_weights/internimage_segmentation.pth
🎯 Use Cases
Mobile-based crop disease diagnosis

Precision agriculture solutions

Dataset quality assurance

Research and industrial deployment

🤝 Contributing
Contributions, issues, and feature requests are welcome.
Please follow standard GitHub contribution guidelines.

📜 License
This project follows the license terms of MMSegmentation and InternImage.

markdown
Copy code

✅ This is a **complete, clean, GitHub-ready README.md** that you can **copy–paste directly**.

If you want next:
- 📌 GitHub **badges**
- 📖 **Citation (BibTeX)**
- 📱 **Mobile-developer-only README**
- 📊 **Swagger API examples**

Just tell me 👍
