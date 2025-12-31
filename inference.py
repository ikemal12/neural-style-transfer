import torch
import time
from PIL import Image
import torchvision.transforms as transforms
from transformer_net import TransformerNetOld
from utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() 
                 if 'running_mean' not in k and 'running_var' not in k}
    model = TransformerNetOld().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def stylize_image(content_path, model_path, output_path):
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)) 
    ])
    content_image = Image.open(content_path)
    content_tensor: torch.Tensor = content_transform(content_image)  # type: ignore
    content_tensor = content_tensor.unsqueeze(0).to(device)
    model = load_model(model_path)
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(content_tensor)
    end_time = time.time()
    output_tensor = output_tensor.squeeze(0).clamp(0, 255).div(255)
    save_image(output_tensor, output_path)
    elapsed = end_time - start_time
    print(f"Stylization complete in {elapsed:.2f} seconds")
    print(f"Result saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', required=True, help='Path to content image')
    parser.add_argument('--model', required=True, help='Path to pretrained model')
    parser.add_argument('--output', required=True, help='Path to save output')
    args = parser.parse_args()
    stylize_image(args.content, args.model, args.output)