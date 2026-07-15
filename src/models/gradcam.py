"""
Grad-CAM Module - Visual Explanations
"""

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(model, input_tensor, target_class):
    target_layer = model.backbone.layer4[-1]
    
    def target_fn(output):
        return output[target_class]
    
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=input_tensor, targets=[target_fn])[0]
    
    return heatmap


def overlay_heatmap(image_array, heatmap, intensity=0.6):
    if heatmap is None:
        return (image_array * 255).astype(np.uint8)
    
    overlay = show_cam_on_image(image_array, heatmap, use_rgb=True).astype(np.float32) / 255.0
    blended = (1 - intensity) * image_array + intensity * overlay
    return np.clip(blended * 255, 0, 255).astype(np.uint8)
