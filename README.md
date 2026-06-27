# Brain-Tumor-Segmentation-Project
# 🧠 Brain Tumor Segmentation Using Swin-UNet

A deep learning-based medical image segmentation system for accurate brain tumor detection from MRI scans using the **Swin-UNet** architecture. The model is trained on the **BraTS 2020** dataset and deployed through a **Streamlit** web application for real-time segmentation and visualization.

---

## 📖 Overview

Brain tumor segmentation plays a crucial role in medical diagnosis and treatment planning. Manual segmentation of MRI scans is time-consuming and prone to human error. This project presents an automated segmentation system based on **Swin-UNet**, a Transformer-powered U-Net architecture that effectively captures both local and global image features.

The project also explores inference optimization techniques to improve deployment efficiency while maintaining high segmentation accuracy.

---

## ✨ Features

- 🧠 Automatic brain tumor segmentation from MRI scans
- 🔬 Swin-UNet Transformer-based architecture
- 📊 Trained on the BraTS 2020 dataset
- 🎯 Binary tumor segmentation
- 🌐 Interactive Streamlit web application
- ⚡ Optimized inference pipeline
- 💾 Pre-trained model included
- 🚀 GPU acceleration support

---

# 🛠 Technologies Used

- Python
- PyTorch
- Swin-UNet
- Streamlit
- OpenCV
- NumPy
- Albumentations
- HDF5
- ONNX Runtime
- TensorRT
- Git LFS

---

# 📂 Project Structure

```
Brain-Tumor-Segmentation-Project/
│
├── app.py                         # Streamlit application
├── backend.py                     # Segmentation backend
├── requirements.txt               # Required Python packages
├── SwinUNet_best.pt               # Trained model
├── demo_images/
│   └── sample_h5_slices/
├── .gitignore
├── .gitattributes
└── README.md
```

---

# 🧠 Model Architecture

The proposed model is based on **Swin-UNet**, a Transformer-based semantic segmentation architecture designed for medical imaging.

### Key Features

- Swin Transformer Encoder
- Window-based Multi-Head Self Attention
- Shifted Window Mechanism
- Hierarchical Feature Learning
- Skip Connections
- CNN-based Decoder

Unlike conventional CNN-based models, Swin-UNet effectively captures both local and long-range contextual information, resulting in improved segmentation performance.

---

# 📊 Dataset

**Dataset:** BraTS 2020 (Brain Tumor Segmentation Challenge)

### MRI Modalities

- T1
- T1ce
- T2
- FLAIR

### Dataset Details

- Pixel-level ground truth masks
- Converted into 2D `.h5` slices
- Binary segmentation (Tumor vs Background)

---

# 🔄 Data Preprocessing

The preprocessing pipeline includes:

- Loading HDF5 MRI slices
- Converting images into HWC format
- Binary mask generation
- Image resizing (256 × 256)
- Pixel normalization

### Data Augmentation

- Horizontal Flip
- Random Affine Transformation

These augmentations improve model robustness and generalization.

---

# 📈 Data Splitting

The dataset was divided as follows:

- **70%** Training
- **15%** Validation
- **15%** Testing

To prevent data leakage, splitting was performed based on **Volume ID** rather than individual slices.

Pure background slices were removed to improve sensitivity toward small tumor regions.

---

# ⚙ Training Configuration

| Parameter | Value |
|-----------|--------|
| Model | Swin-UNet |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Mixed Precision | FP16 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Enabled |
| Memory Format | Channels Last |

---

# 🎯 Hybrid Loss Function

The model is trained using a **hybrid loss function** combining **Focal Loss** and **Tversky Loss**.

```
Loss = 0.4 × Focal Loss + 0.6 × Tversky Loss
```

### Focal Loss

Focal Loss focuses on difficult-to-classify pixels while reducing the influence of easy background pixels. This improves learning for highly imbalanced medical datasets.

### Tversky Loss

Tversky Loss is specifically designed for medical image segmentation. It provides better control over false positives and false negatives, making it highly effective for detecting small tumor regions.

### Advantages

- Handles severe class imbalance
- Improves segmentation of small tumors
- Reduces false negatives
- Produces accurate tumor boundaries
- Enhances overall segmentation performance

---

# 📊 Evaluation Metric

Model performance is evaluated using the **Dice Coefficient**.

```
Dice = (2 × Intersection)
       -------------------------
       Prediction + Ground Truth
```

Higher Dice scores indicate better overlap between predicted and ground truth masks.

---

# ⚡ Inference Optimization

To improve deployment efficiency, the trained model was evaluated using multiple inference frameworks.

- PyTorch
- TorchScript
- ONNX Runtime
- TensorRT (FP16)

The objective was to reduce inference latency and memory usage while maintaining segmentation accuracy.

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/Niya2201/Brain-Tumor-Segmentation-Project.git
```

Move into the project directory

```bash
cd Brain-Tumor-Segmentation-Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

Upload an MRI `.h5` slice to generate the predicted tumor segmentation.

---

# 📌 Future Enhancements

- Multi-class brain tumor segmentation
- 3D Swin-UNETR implementation
- DICOM image support
- Cloud deployment
- Explainable AI (Grad-CAM)
- Clinical integration
- Real-time hospital deployment

---

# 👨‍💻 Team Members

- **Keerthi Krishna G.**
- **Krishnapriya A. L.**
- **Niya Aniyan**

### Guide

**Prof. Janisha A**

---

# 📄 License

This project was developed as part of the B.Tech Computer Science and Engineering final-year academic project.

The repository is intended for educational and research purposes.
