import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from p2_models.models import *


def interpret_with_GradCAM(model, input_img, device) -> np.ndarray:
    """
    Plots the most important pixels of an image for the model prediction.
    """
    input_tensor = model.get_infer_transforms()(input_img)
    input_tensor = input_tensor.to(device).unsqueeze(0)

    cam = GradCAM(model=model, target_layers=model.get_target_layers())
    targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = input_tensor[0].permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    return visualization
