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
- **Automatic 3D/4D mode selection** via `NeuroEncoder` class:  
  - Supports both 3D and 4D fMRI input structures  
  - 3D Mode: uses `3DResNet` or `3DViT`.  
  - 4D Mode: uses `TemporalTransformer` with a `ProjectionHead`.  
  - For best 4D performance, **[Swin4D](https://github.com/gillet-thomas/SWIN)** is recommended.  
- **Supported datasets**:  
  - ADNI (Alzheimer’s Disease Neuroimaging Initiative) in both 3D and 4D 
  - In-house Pain fMRI dataset in 3D on the timepoints
- **Explainability tools**:  
  - G3DViT for 3D ViT Grad-CAM visualizations  
  - pytorch_grad_cam Grad-CAM / LayerCAM / GradCAM-EW for 3D ResNet  
  - SHAP and Captum for 3D ResNet

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
│ │ ├── NeuroEncoder.py # Main fMRI encoder (3D or 4D)
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
python main.py "run_name" --cuda 0
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

> **Note for 4D Training**:
> When switching to 4D fMRI training, update the data loading loop in `Trainer.py` to remove the timepoint dimension from the dataloader unpacking. Replace: `for i, (subject, timepoint, fMRI, group, gender, age, age_group)` with: `for i, (subject, fMRI, group, gender, age, age_group)`  
> This ensures compatibility with the 4D dataset format where all timepoints are processed as a single volume sequence.


### Inference and model explainability
```
python main.py --inference
```

For explainability, run the scripts in the `explainability/` directory:

- Visualize **Grad-CAM** attention maps for 3DViT setup using [G3D-ViT](https://github.com/gillet-thomas/G3DViT)
- Compute **SHAP** or **Captum** explanations for 3DResNet setup  
 > **Note**: To avoid path issues, move the desired explainability script (e.g., gradcam3DViT_fmris.py) to the root directory of the project before running it.
---


## 📊 Results & Observations

| Dataset             | Task                                | Validation Accuracy (%) |
|----------------------|-------------------------------------|---------------|
| **ADNI**             | Age group (Young vs Old, Q1–Q4)     | **95.23**     |
|                      | Gender                              | **93.20**     |
|                      | Alzheimer’s vs Control (AD vs CN)   | **97.72**     |
|                      | Multi-target (Age + Gender)         | **89.20**     |
| **ADNI Pain Dataset**| Age                                 | **89.61**     |
|                      | Gender                              | **93.57**     |

**Insights:**
- Grad-CAM maps for age and gender often highlight structural rather than functional regions.  
- Pain-related prediction remains challenging—likely due to subtle or noisy patterns in resting-state data.  
- Hyperparameter sweeps identified configs >80% validation accuracy, but not consistently confirmed by k-fold.

---

## Conclusion

**NeuroViT** provides a flexible transformer-based framework for encoding brain imaging data:

- Performs well on **3D fMRI/MRI timepoint encoding**  
- Experimental **4D temporal encoding** (with custom Temporal Transformer)  
- For best 4D performance, use **[Swin4D](https://github.com/gillet-thomas/SWIN)**
