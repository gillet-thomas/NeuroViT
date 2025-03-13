import requests
from PIL import Image
import open_clip
import torch
import yaml
import numpy as np
import warnings
import nibabel as nib
from legrad import LeWrapper, LePreprocess, visualize
from src.fmriEncoder import fmriEncoder
from nilearn.image import load_img

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Config
BASE_PATH = "/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/"
FMRI_PATH = "/mnt/data/iai/datasets/fMRI_marian/151/wau4D.nii"
config = yaml.safe_load(open(BASE_PATH + "configs/config.yaml"))
config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load Model and GradCAM
model = fmriEncoder(config).to(config["device"]).eval()
model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)

fmri_img = load_img(FMRI_PATH)
fmri_data = fmri_img.get_fdata(dtype=np.float32)                # Shape: (91, 109, 91, 146)
fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 90)
fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalize
input_tensor = torch.tensor(fmri_norm).to(config["device"])
fmri_slice = fmri_norm[:, :, 45]  # Choose middle slice
fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)
fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
fmri_rgb = torch.Tensor(fmri_rgb.transpose(2, 0, 1)).unsqueeze(0)
print(fmri_rgb.shape, "fmri_rgb")

# ------- model's paramters -------
model_name = 'ViT-B-16'
pretrained = 'laion2b_s34b_b88k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------- init model -------
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name=model_name)
model.eval()
# ------- Equip the model with LeGrad -------
model = LeWrapper(model)
# ___ (Optional): Wrapper for Higher-Res input image ___
preprocess = LePreprocess(preprocess=preprocess, image_size=448)

# ------- init inputs: image + text -------
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device) # 1, 3, 448, 448
text = tokenizer(['a photo of a cat', 'a photo of a remote control']).to(device)

# -------
text_embedding = model.encode_text(text, normalize=True)
print(image.shape)
explainability_map = model.compute_legrad_clip(image=image, text_embedding=text_embedding)

# ___ (Optional): Visualize overlay of the image + heatmap ___
visualize(heatmaps=explainability_map, image=image, save_path=True)