"""
MedCLIP Module - Medical Vision-Language Model
"""

import torch
import numpy as np

try:
    from medclip import MedCLIPModel, MedCLIPProcessor
    MEDCLIP_AVAILABLE = True
except ImportError:
    MEDCLIP_AVAILABLE = False


class MedCLIPExplainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.processor = None
        self.loaded = False
        if MEDCLIP_AVAILABLE:
            self.load()
    
    def load(self):
        if not MEDCLIP_AVAILABLE:
            return False
        try:
            self.model = MedCLIPModel.from_pretrained("medclip-vit-base")
            self.processor = MedCLIPProcessor.from_pretrained("medclip-vit-base")
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            return True
        except:
            return False
    
    def get_similarity(self, image, text_prompts):
        if not self.loaded:
            return None
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits_per_image.cpu().numpy()
    
    def get_best_prompt(self, image, text_prompts):
        similarities = self.get_similarity(image, text_prompts)
        if similarities is None:
            return None, None
        best_idx = int(np.argmax(similarities[0]))
        return text_prompts[best_idx], float(similarities[0][best_idx])
