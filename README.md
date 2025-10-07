<div align="center">

#  NeuroViT: Vision Transformer-Based Encoding of Neural Data

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9.6-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>

</div>

---

## 📌 Introduction
**NeuroViT** is a **Vision Transformer (ViT)-based framework** for encoding **3D and 4D neuroimaging data**, such as fMRI and MRI volumes.  
It supports both **3D timepoint-level encoding** and **4D spatio-temporal encoding** of fMRI sequences.  

While **NeuroViT** achieves **strong and stable performance on 3D data**,  
for full **4D temporal modeling**, the **[Swin4D](https://github.com/gillet-thomas/SWIN)** encoder is recommended.

The project also includes **G3DViT**, a 3D Grad-CAM module for model interpretability, enabling visualization of brain regions driving model predictions.  

> Note: **NeuroViT** is part of a multimodal research effort to create a shared embedding space combining fMRI data, behavioral measures, and blood-based biomarkers. Its goal is to classify patients based on **pain sensitivity** while also analyzing demographic factors like age and gender.

---

## 🚀 Key Features

- **3D Vision Transformer (ViT)** – Encodes single fMRI/MRI volumes (timepoints).  
- **3D ResNet** – Baseline convolutional encoder for 3D volumes.  
- **4D Encoder (Temporal Transformer + Projection Head)** – Combines spatial and temporal features for 4D fMRI sequences.  
- **G3DViT Grad-CAM** – 3D Grad-CAM visual explanations for both ResNet and ViT models.  
- **Automatic 3D/4D mode selection** via `fmriEncoder` class:  
  - Supports both 3D and 4D fMRI input structures  
  - 3D Mode: uses `3DResNet` or `3DViT`.  
  - 4D Mode: uses `TemporalTransformer` with a `ProjectionHead`.  
  - For best 4D performance, **[Swin4D](https://github.com/gillet-thomas/SWIN)** is recommended.  
- **Supported datasets**:  
  - ADNI (Alzheimer’s Disease Neuroimaging Initiative) in both 3D and 4D 
  - In-house Pain fMRI dataset in 3D on the timepoints
- **Explainability tools**:  
  - Grad-CAM / LayerCAM / GradCAM-EW for 3D ResNet  
  - G3DViT for 3D ViT Grad-CAM visualizations  
  - SHAP and Captum integration for feature-level interpretability

---

## 📁 Project Structure

```
NeuroViT/
├── configs/
│ ├── config.yaml # Main configuration for 3D models
│ ├── config4D.yaml # Configuration for 4D model training
│ └── sweep.yaml # Hyperparameter sweep configuration for WandB
├── explainability/
│ ├── gradcam3DViT_fmris.py # G3D-ViT for 3D Vision Transformer
│ ├── xAi_gradcam_Resnet3D/ # pytorch_grad_cam for 3D ResNet
│ ├── xAi_gradcam_ViT3D/ # pytorch_grad_cam - Grad-CAM for 3D Vision Transformer
│ ├── xAi_captum_Resnet3D/ # Captum-based explainability
│ └── xAi_shap_Resnet3D/ # SHAP explainability
├── src/
│ ├── data/
│ │ ├── DatasetADNI.py # ADNI dataset loader
│ │ ├── DatasetPain.py # Pain dataset loader
│ │ └── correlation.py # Correlation analysis
│ ├── models/
│ │ ├── fmriEncoder.py # Main fMRI encoder (3D or 4D)
│ │ ├── resnet_3d.py # 3D ResNet implementation
│ │ └── vit_3d.py # 3D Vision Transformer
│ └── Trainer.py # Training loop and utilities
├── results/ # Output results and checkpoints
└── main.py # Main entry point
```


---

## 💻 Getting Started

## Train

4D mode supports **gradient accumulation** for larger virtual batch sizes.  
You can run **WandB sweeps** to explore multiple hyperparameter configurations.

```bash
python main.py "run_name" --inference
```

Edit the desired configuration file in `configs/`:
- `config.yaml` → 3D training  
- `config4D.yaml` → 4D training  
- `sweep.yaml` → WandB hyperparameter sweeps  

Runtime flags (from `main.py`):
- `name` → WandB run name  
- `--cuda` → GPU index  
- `--wandb` → Enable/disable logging  
- `--inference` → Skip training and run retrieval  
- `--sweep` → Run WandB sweep  

### Inference and model explainability
```
python main.py "run_name" --cuda 0 --inference --wandb false
```

For explainability, run the scripts in the `explainability/` directory:

- Visualize **Grad-CAM** attention maps for 3DViT setup using [G3D-ViT](https://github.com/gillet-thomas/G3DViT)
- Compute **SHAP** or **Captum** explanations for 3DResNet setup  

---


## 📈 Results & Observations

| Task | Model | Performance | Notes |
|------|--------|-------------|-------|
| Age & Gender Classification | 3D ResNet / 3D ViT | ✅ Strong, consistent (k-fold validated) | Structural cues (ventricles, cortex thickness) seem most influential |
| Pain Prediction | 3D ResNet | ⚠️ ~50% accuracy | Limited discriminative power |
| Pain Prediction | 3D ViT | ⭐ 73–77% accuracy | Promising but possibly overfitted (lucky split) |

**Insights:**
- Grad-CAM maps for age and gender often highlight structural rather than functional regions.  
- Pain-related prediction remains challenging—likely due to subtle or noisy patterns in resting-state data.  
- Hyperparameter sweeps identified configs >80% validation accuracy, but not consistently confirmed by k-fold.

---

## Conclusion

**NeuroViT** provides a flexible transformer-based framework for encoding brain imaging data:

- Performs well on **3D fMRI/MRI timepoint encoding**  
- Experimental **4D temporal encoding** (with Temporal Transformer)  
- For best 4D performance, use **[Swin4D](https://github.com/gillet-thomas/SWIN)**
