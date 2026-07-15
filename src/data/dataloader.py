"""
Data Loader for DETECCION DE FRACTURAS Dataset
Handles loading COCO annotations, image preprocessing, and PyTorch DataLoader creation.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path


class FractureDataset(Dataset):
    """PyTorch Dataset for wrist fracture X-ray images."""
    
    def __init__(self, img_dir, annotation_file, transform=None):
        """
        Args:
            img_dir: Directory with all the images
            annotation_file: Path to COCO JSON annotation file
            transform: Optional transform to be applied on a sample
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Map image_id to file name
        self.images = self.coco_data['images']
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.images}
        
        # Get class info
        self.categories = self.coco_data['categories']
        self.class_names = {cat['id']: cat['name'] for cat in self.categories}
        self.class_to_idx = {cat['name']: cat['id'] for cat in self.categories}
        
        # Build image to annotations mapping
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Create list of valid image IDs (images with at least one annotation)
        self.valid_image_ids = [img['id'] for img in self.images 
                               if img['id'] in self.image_annotations]
        
        print(f"✅ Loaded {len(self.valid_image_ids)} images with annotations")
        print(f"✅ Classes: {self.class_names}")