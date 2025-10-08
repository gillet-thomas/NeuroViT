**activation_map** folder (architecture 3D Resnet, pytorch_grad_cam)
files: original tutorial from pytorch_grad_cam documentation

**xAi_captum** folder (architecture 3D Resnet, ageGroup classification, captum)
files: captum_LayerCAM.py (**works well**), captum_IntegratedGradients.py (not working), captum_IntegratedGradients2.py (results not accurate)

**xAi_gradcam** folder (architecture 3D Resnet, ageGroup classification, pytorch_grad_cam)
files: gradcam.py (**works well** for both ageGroup and gender classification), gradcam_sMRI.py (**works well** for classification on sMRI of Marian)

**xAi_shap** folder (architecture 3D Resnet, ageGroup classification, shap)
files: shape.py (never managed to make it work)