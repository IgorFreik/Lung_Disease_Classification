import gradio as gr
import torch
from torchvision import transforms
from p3_analysis.interpretations import interpret_with_GradCAM

# Import model
from p2_models.models import *

LABELS = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']


def predict(inp, model, device):
    inp = transforms.ToTensor()(inp).to(device).unsqueeze(0).view(1, 3, 128, 128)  # [1, 3, 128, 128]
    with torch.no_grad():
        output = model(inp)  # [1, num_diseases]
        predictions = output.cpu().detach().numpy()[0]

    return {label: float(prob) for label, prob in zip(LABELS, predictions)}


# def create_interface(model, device):
#     # model = BaselineSimple().to(device)
#     # model.load_state_dict(torch.load('model_weights/best_checkpoint.model'))
#
#     predict = lambda inp: forward(inp, model, device)
#
#     gr.Interface(fn=predict,
#                  inputs=gr.Image(type="pil"),
#                  outputs=gr.Label()).launch(share=True)


def show_web_interface(model, device):
    """
    Runs a gradio web interface for the model.
    :param model: a LeafCounter class instance.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Leaf counter with UNet++ and ResNet50")

        with gr.Row():
            im = gr.Image()
            txt = gr.Textbox(value="", label="Output")

        predict_fn = lambda img: predict(model, img, device), interpret_with_GradCAM(model,
                                                                                     model.get_target_layers(),
                                                                                     im, device)

        btn = gr.Button(value="Count leafs")
        btn.click(predict_fn, inputs=[im], outputs=[txt, im])

        gr.Markdown("## Image Examples")
        #
        # gr.Examples(
        #     examples=["../examples/leaf_im_big.png"],
        #     inputs=im,
        #     outputs=txt,
        #     fn=predict_fn
        # )
    demo.launch()
