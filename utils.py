import os
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime

unloader = transforms.ToPILImage()

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    return image

def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = tensor_to_pil(tensor)
    image.save(path)

def save_comparison(content_image, style_image, output_image, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    content_display = content_image.cpu().clone().squeeze(0)
    content_display = unloader(content_display)
    axes[0].imshow(content_display)
    axes[0].set_title('Content Image')
    axes[0].axis('off')

    style_display = style_image.cpu().clone().squeeze(0)
    style_display = unloader(style_display)
    axes[1].imshow(style_display)
    axes[1].set_title('Style Image')
    axes[1].axis('off')

    output_display = output_image.cpu().clone().squeeze(0)
    output_display = unloader(output_display)
    axes[2].imshow(output_display)
    axes[2].set_title('Output Image')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def create_output_directory(base_dir, content_name, style_name):
    content_basename = os.path.splitext(os.path.basename(content_name))[0]
    style_basename = os.path.splitext(os.path.basename(style_name))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{content_basename}_styled_with_{style_basename}_{timestamp}"
    output_dir = os.path.join(base_dir, dir_name)   
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
