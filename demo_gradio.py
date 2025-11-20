import gradio as gr
import os
import numpy as np
from PIL import Image
import torch
import copy
from omegaconf import OmegaConf
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import hf_hub_download

from chord import ChordModel
from chord.module import make
from chord.util import get_positions, rgb_to_srgb

EXAMPLES_USECASE_1 = [
    [f"examples/generated/{f}"] 
    for f in sorted(os.listdir("examples/generated"))
]
EXAMPLES_USECASE_2 = [
    [f"examples/in_the_wild/{f}"] 
    for f in sorted(os.listdir("examples/in_the_wild"))
]
EXAMPLES_USECASE_3 = [
    [f"examples/specular/{f}"] 
    for f in sorted(os.listdir("examples/specular"))
]

MODEL_OBJ = None
MODEL_CKPT_PATH = hf_hub_download(repo_id="Ubisoft/ubisoft-laforge-chord", filename="chord_v1.ckpt")
def load_model(ckpt_path):
    print("Loading model from:", ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load("config/chord.yaml")
    model = ChordModel(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model

def run_model(model, img: Image.Image):
    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    image = to_tensor(img).to(next(model.parameters()).device)
    x = v2.Resize(size=(1024, 1024), antialias=True)(image).unsqueeze(0)
    with torch.no_grad() as no_grad, torch.autocast(device_type="cuda") as amp:
        output = model(x)
    output.update({"input": image}) 
    return output

def relit(model, maps):
    maps['metallic'] = maps.get('metalness', torch.zeros_like(maps['basecolor']))
    device = next(model.parameters()).device
    h, w = maps["basecolor"].shape[-2:]
    light = make("point-light", {"position": [0, 0, 10]}).to(device)
    pos = get_positions(h, w, 10).to(device)
    camera = torch.tensor([0, 0, 10.0]).to(device)
    for key in maps:
        if maps[key].dim() == 3:
            maps[key] = maps[key].unsqueeze(0)
        maps[key] = maps[key].permute(0,2,3,1)  # BxCxHxW -> BxHxWxC
    rgb = model.model.compute_render(maps, camera, pos, light).squeeze(0).permute(0,3,1,2)  # GxBxHxWxC -> BxCxHxW
    return torch.clamp(rgb_to_srgb(rgb), 0, 1)

def inference(img):
    global MODEL_OBJ

    if MODEL_OBJ is None or getattr(MODEL_OBJ, "_ckpt", None) != MODEL_CKPT_PATH:
        MODEL_OBJ = load_model(MODEL_CKPT_PATH)
        MODEL_OBJ._ckpt = MODEL_CKPT_PATH  # store path inside object

    if img is None:
        return None, None, None, None, None
    
    ori_h, ori_w = img.size[1], img.size[0]
    out = run_model(MODEL_OBJ, img)
    maps = copy.deepcopy(out)
    rendered = relit(MODEL_OBJ, maps)
    resize_back = v2.Resize(size=(ori_h, ori_w), antialias=True)
    return (
        to_pil_image(resize_back(out["basecolor"]).squeeze(0)),
        to_pil_image(resize_back(out["normal"]).squeeze(0)),
        to_pil_image(resize_back(out["roughness"]).squeeze(0)),
        to_pil_image(resize_back(out["metalness"]).squeeze(0)),
        to_pil_image(resize_back(rendered).squeeze(0)),
    )

with gr.Blocks(title="Chord") as demo:

    gr.Markdown("# **Chord: Chain of Rendering Decomposition for PBR Material Estimation from Generated Texture Images**")
    gr.Markdown("Upload an image or select an example to estimate PBR channels.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image", height=512)

            gr.Markdown("### Example Inputs — Generated Textures")
            gr.Examples(
                examples=EXAMPLES_USECASE_1,
                inputs=[input_img],
                label="Examples (Generated Textures)"
            )

            gr.Markdown("### Example Inputs — In The Wild Photographs")
            gr.Examples(
                examples=EXAMPLES_USECASE_2,
                inputs=[input_img],
                label="Examples (In The Wild Photographs)"
            )

            gr.Markdown("### Example Inputs — Specular Textures")
            gr.Examples(
                examples=EXAMPLES_USECASE_3,
                inputs=[input_img],
                label="Examples (Specular Textures)"
            )

            run_button = gr.Button("Run Estimation")

        with gr.Column():
            gr.Markdown("### Predicted Channels")
            basecolor_out = gr.Image(label="Basecolor", height=512)
            normal_out = gr.Image(label="Normal", height=512)
            roughness_out = gr.Image(label="Roughness", height=512)
            metallic_out = gr.Image(label="Metalness", height=512)

            gr.Markdown("### Relit Output")
            render_out = gr.Image(label="Relit Image (Centered Point Light)", height=512)

    run_button.click(
        inference,
        inputs=[input_img],
        outputs=[basecolor_out, normal_out, roughness_out, metallic_out, render_out]
    )

if __name__ == "__main__":
    demo.launch()
