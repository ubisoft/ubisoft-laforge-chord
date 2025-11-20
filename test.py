import sys
import os
import torch
from torchvision.transforms import v2
import typer
from pathlib import Path
from typing_extensions import Annotated
from omegaconf import OmegaConf
import tqdm
from huggingface_hub import hf_hub_download

from chord import ChordModel
from chord.io import read_image, save_maps

app = typer.Typer(pretty_exceptions_show_locals=False)

def setup_python_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

def get_image_files(indir):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in Path(indir).rglob(ext):
            files.append(str(path))
    return files

@app.command()
def inference(
    input_dir: Annotated[str, typer.Option(default=..., help="Paths to the input image directory.")],
    output_dir: Annotated[str, typer.Option(help="Directory to save output maps.")] = "output",
    config_path: Annotated[str, typer.Option(help="Path to the config file.")] = "config/chord.yaml",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)
    model = ChordModel(config)
    ckpt_path = hf_hub_download(repo_id="Ubisoft/ubisoft-laforge-chord", filename="chord_v1.ckpt")
    print(f"[INFO] Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    image_files = get_image_files(input_dir)
    print(f"[INFO] found {len(image_files)} images in {input_dir}")
    print(f"[INFO] saving results to {output_dir}")

    for image_file in tqdm.tqdm(image_files, desc="[INFO] processing images"):
        image = read_image(image_file).to(device)
        ori_h, ori_w = image.shape[-2:]
        x = v2.Resize(size=(1024, 1024), antialias=True)(image).unsqueeze(0)
        image_name = Path(image_file).stem
        with torch.no_grad() as no_grad, torch.autocast(device_type="cuda") as amp:
            output = model(x)
        for key in output.keys():
            output[key] = v2.Resize(size=(ori_h, ori_w), antialias=True)(output[key])
        output.update({"input": image})
        save_maps(os.path.join(output_dir, image_name), output)

if __name__ == "__main__":
    setup_python_path()
    app()