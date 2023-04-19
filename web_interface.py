import gradio as gr
import torch
from torchvision import transforms
from p3_analysis.interpretations import interpret_with_GradCAM

# Import model
from p2_models.models import *

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
          'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


def predict(model, inp, device):
    transform = model.get_infer_transforms()
    inp_tensor = transform(inp).to(device).unsqueeze(0)

    with torch.no_grad():
        output = model(inp_tensor)  # [1, num_diseases]
        predictions = output.cpu().detach().numpy()[0]

    return {label: float(prob) for label, prob in zip(LABELS, predictions)}


def show_web_interface(model, device):
    """
    Runs a gradio web interface for the model.
    :param model: a LeafCounter class instance.
    :param device: device: 'cuda'/'cpu'.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Lung Disease Classification")

        with gr.Row():
            im = gr.Image()
            lbl = gr.Label()

        predict_fn = lambda img: (predict(model, img, device),
                                  interpret_with_GradCAM(model, img, device))

        btn = gr.Button(value="Get predictions")
        btn.click(predict_fn, inputs=[im], outputs=[lbl, im])

        gr.Markdown("## Image Examples")

        gr.Examples(
            examples=["data/images/00000001_000.png"],
            inputs=im,
            outputs=lbl,
            fn=predict_fn
        )
    demo.launch()
