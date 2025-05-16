import os
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
from datetime import datetime
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from src.fmriEncoder import fmriEncoder
from src.data.DatasetGradCAM import GradCAMDataset


def create_itksnap_workspace(original_nifti, gradcam_nifti, output_dir):
    # Create the root element
    registry = ET.Element("registry")
    
    # Add basic entries
    ET.SubElement(registry, "entry", key="SaveLocation", value=output_dir)
    ET.SubElement(registry, "entry", key="Version", value=datetime.now().strftime("%Y%m%d"))
    
    # Add annotations
    annotations = ET.SubElement(registry, "folder", key="Annotations")
    ET.SubElement(annotations, "entry", key="Format", value="ITK-SNAP Annotation File")
    ET.SubElement(annotations, "entry", key="FormatDate", value="20150624")
    
    # Add layers
    layers = ET.SubElement(registry, "folder", key="Layers")
    
    # Main layer (original image)
    layer0 = ET.SubElement(layers, "folder", key="Layer[000]")
    ET.SubElement(layer0, "entry", key="AbsolutePath", value=original_nifti)
    ET.SubElement(layer0, "entry", key="Role", value="MainRole")
    ET.SubElement(layer0, "entry", key="Tags", value="")
    
    # Layer metadata for main image
    layer_meta0 = ET.SubElement(layer0, "folder", key="LayerMetaData")
    ET.SubElement(layer_meta0, "entry", key="Alpha", value="255")
    ET.SubElement(layer_meta0, "entry", key="CustomNickName", value="")
    ET.SubElement(layer_meta0, "entry", key="Sticky", value="0")
    
    # Display mapping for main image
    display_mapping0 = ET.SubElement(layer_meta0, "folder", key="DisplayMapping")
    color_map0 = ET.SubElement(display_mapping0, "folder", key="ColorMap")
    ET.SubElement(color_map0, "entry", key="Preset", value="Grayscale")
    
    # Overlay layer (gradcam)
    layer1 = ET.SubElement(layers, "folder", key="Layer[001]")
    ET.SubElement(layer1, "entry", key="AbsolutePath", value=gradcam_nifti)
    ET.SubElement(layer1, "entry", key="Role", value="OverlayRole")
    ET.SubElement(layer1, "entry", key="Tags", value="")
    
    # Layer metadata for overlay
    layer_meta1 = ET.SubElement(layer1, "folder", key="LayerMetaData")
    ET.SubElement(layer_meta1, "entry", key="Alpha", value="0.5")
    ET.SubElement(layer_meta1, "entry", key="CustomNickName", value="")
    ET.SubElement(layer_meta1, "entry", key="Sticky", value="1")
    
    # Display mapping for overlay
    display_mapping1 = ET.SubElement(layer_meta1, "folder", key="DisplayMapping")
    color_map1 = ET.SubElement(display_mapping1, "folder", key="ColorMap")
    ET.SubElement(color_map1, "entry", key="Preset", value="Jet")
    
    # Add TimePointProperties
    time_props = ET.SubElement(registry, "folder", key="TimePointProperties")
    ET.SubElement(time_props, "entry", key="FormatVersion", value="1")
    
    time_points = ET.SubElement(time_props, "folder", key="TimePoints")
    ET.SubElement(time_points, "entry", key="ArraySize", value="1")
    
    time_point = ET.SubElement(time_points, "folder", key="TimePoint[1]")
    ET.SubElement(time_point, "entry", key="Nickname", value="")
    ET.SubElement(time_point, "entry", key="Tags", value="")
    ET.SubElement(time_point, "entry", key="TimePoint", value="1")
    
    # Create the XML tree
    tree = ET.ElementTree(registry)
    
    # Add XML declaration and doctype
    xml_str = '<?xml version="1.0" encoding="UTF-8" ?>\n'
    xml_str += '<!--ITK-SNAP (itksnap.org) Project File-->\n\n'
    xml_str += '<!DOCTYPE registry [\n'
    xml_str += '<!ELEMENT registry (entry*,folder*)>\n'
    xml_str += '<!ELEMENT folder (entry*,folder*)>\n'
    xml_str += '<!ELEMENT entry EMPTY>\n'
    xml_str += '<!ATTLIST folder key CDATA #REQUIRED>\n'
    xml_str += '<!ATTLIST entry key CDATA #REQUIRED>\n'
    xml_str += '<!ATTLIST entry value CDATA #REQUIRED>\n'
    xml_str += ']>\n'
    xml_str += ET.tostring(registry, encoding='unicode')
    
    # Save to file
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(original_nifti))[0] + ".itksnap")
    with open(output_path, 'w') as f:
        f.write(xml_str)
    
    return output_path

def get_sample_gradcam(id, save_sample_attention=False):

    sample = dataset[id]
    input_tensor = sample[0].to(config["device"]).unsqueeze(0)
    print(f"ID: {id} - Label: {sample[1].item()}, Coordinates: {sample[2].tolist()}")

    cube_size = config["cube_size"]
    patch_x = int(sample[2][0] + cube_size // 2)
    patch_y = int(sample[2][1] + cube_size // 2)
    patch_z = int(sample[2][2] + cube_size // 2)
    # print(f"Patch coordinates: {patch_x}, {patch_y}, {patch_z}")

    # Get attention map
    attention_map, class_idx = model.get_attention_map(input_tensor)        # output [90, 90, 90]
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=0, slice_idx=patch_x)

    if save_sample_attention:
        nib.save(nib.Nifti1Image(attention_map.cpu().numpy(), np.eye(4)), f'{config["base_path"]}/{id}_gradcam_3dd.nii')
        save_gradcam_3d(attention_map, id, sample)
    
    return id, img, attn, class_idx, sample[1]

def create_gradcam_plot(save_sample_attention=False):
    
    results = [get_sample_gradcam(id, save_sample_attention=save_sample_attention) for id in ids]

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle(f'DatasetGradCAM {config["grid_size"]}grid {config["cube_size"]}cube {config["vit_patch_size"]}patch {config["grid_noise"]}noise', fontsize=16)
    
    # Plot each subject's results
    for idx, (ID, image, attention, class_idx, sample) in enumerate(results):
        row = idx // cols
        col = idx % cols
        ax = axes[col] if rows == 1 else axes[row, col]
        
        ax.imshow(-image+1 if config["grid_noise"] < 1 else image, cmap='gray')    # INVERSE BRIGHTNESS
        heatmap = ax.imshow(attention, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Subject {ID} (Class {class_idx.item()})')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        (axes[col] if rows == 1 else axes[row, col]).axis('off')

    # file_name = f'DatasetGradCAM_{config["grid_size"]}grid_{config["cube_size"]}cube_{config["vit_patch_size"]}patch_{config["grid_noise"]}noise_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'.replace('.', 'p')
    file_name = f'DatasetGradCAM_{config["grid_size"]}grid_{config["cube_size"]}cube_{config["vit_patch_size"]}patch_{config["grid_noise"]}noise'.replace('.', 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(config["base_path"], f"explainability/xAi_gradcam3DViT/DatasetGradCAM/{file_name}.png"), dpi=300)
    plt.close()
    print(f"All results saved to {file_name}.png")

def save_gradcam_3d(attention_map, id, sample):
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    threshold = 0.2
    coords = np.argwhere(attention_map.cpu().numpy() > threshold)
    values = attention_map.cpu().numpy()[attention_map.cpu().numpy() > threshold]

    # Scatter plot for the regions with attention above the threshold
    if coords.size > 0:
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap='jet', marker='s', alpha=0.6, s=50)
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Attention Value')
    else:
        print(f"No attention values above threshold {threshold} for sample {id}")

    # Add a bounding box for the volume
    ax.set(xlim=(0, attention_map.shape[0]), ylim=(0, attention_map.shape[1]), zlim=(0, attention_map.shape[2])) # Grid size
    ax.set(xlabel='X axis', ylabel='Y axis', zlabel='Z axis')

    # Save figure and nifti file
    save_path = './explainability/xAi_gradcam3DViT/DatasetGradCAM/'
    file_name = f'DatasetGradCAM_{config["grid_size"]}grid_{config["cube_size"]}cube_sample_{config["grid_noise"]}noised_{id}_3D'.replace('.', 'p')
    plt.title(f'3D GradCAM (Label: {sample[1].item()}, coordinates: {sample[2].tolist()})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{file_name}.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    # Config 
    warnings.simplefilter(action='ignore', category=FutureWarning)
    config = yaml.safe_load(open("/mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/fMRI2Vec/configs/config.yaml"))
    config["device"] = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Load Model and Dataset
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)
    dataset = GradCAMDataset(config, mode="val", generate_data=False)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    create_gradcam_plot(save_sample_attention=config["save_gradcam_attention"])
    
