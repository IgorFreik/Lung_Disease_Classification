import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import *


def interpret_with_GradCAM(model, target_layers, input_img, device) -> np.ndarray:
    """
    Plots the most important pixels of an image for the model prediction.
    """
    if not target_layers and isinstance(model, BaselineSimple):
        # This is the target layer for ResNet18 model
        target_layers = [model.layer4[-1]]
    else:
        raise NotImplemented

    input_tensor = input_img.to(device).unsqueeze(0)
    input_img = input_img.numpy()

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.repeat(input_img, 3, axis=0).reshape(128, 128, 3)
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    return visualization
