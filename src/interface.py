import gradio as gr
import torch
from torchvision import transforms

# Import model
from models import *

LABELS = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']


def forward(inp, model, device):
    inp = transforms.ToTensor()(inp).to(device).unsqueeze(0).view(1, 3, 128, 128)  # [1, 3, 128, 128]
    with torch.no_grad():
        output = model(inp)  # [1, num_diseases]
        predictions = output.cpu().detach().numpy()[0]

    return {label: float(prob) for label, prob in zip(LABELS, predictions)}


def create_interface(model, device):
    # model = BaselineSimple().to(device)
    # model.load_state_dict(torch.load('model_weights/best_checkpoint.model'))

    predict = lambda inp: forward(inp, model, device)

    gr.Interface(fn=predict,
                 inputs=gr.Image(type="pil"),
                 outputs=gr.Label()).launch(share=True)
