"""
Grad-CAM: Gradient-weighted Class Activation Mapping
Model'in karar verirken hangi bölgelere odaklandığını gösterir.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM implementation

    Kullanım:
        gradcam = GradCAM(model, target_layer)
        cam, predicted_class = gradcam.generate(input_tensor)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: nn.Module (örn: model.layer4[-1].conv2)
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Grad-CAM backward hooks can fail with inplace ReLU ops.
        # Turn them off once here so evaluation never crashes on VGG/ResNet blocks.
        self._disable_inplace_relu(self.model)

        # Forward hook: activations kaydet
        target_layer.register_forward_hook(self._forward_hook)

        # Backward hook: gradients kaydet
        target_layer.register_full_backward_hook(self._backward_hook)

    def _disable_inplace_relu(self, module):
        for m in module.modules():
            if isinstance(m, nn.ReLU) and m.inplace:
                m.inplace = False

    def _forward_hook(self, module, input, output):
        """Forward pass'ta activation'ları kaydet"""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Backward pass'ta gradient'leri kaydet"""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Grad-CAM haritası oluştur

        Args:
            input_tensor: torch.Tensor shape (1, 3, H, W)
            class_idx: Hangi sınıf için CAM? None = predicted class

        Returns:
            cam: numpy array shape (H, W), normalized [0, 1]
            predicted_class: int
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Predicted class (eğer belirtilmediyse)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass: Gradients compute et
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # CAM computation
        # weights = mean gradient across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # CAM = sum over channels: weights * activations
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze()

        # CPU'ya taşı ve numpy'ye çevir
        cam = cam.cpu().numpy()

        # Normalize: [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, class_idx
