import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128  # size of output image
print(imsize)

loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

def img_loader(img_name):
    img = Image.open(img_name)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)

img_dir = "images/"
content_img = img_loader(img_dir + "tajmahal.jpg")
style_img = img_loader(img_dir + "tree.jpg")

assert content_img.size() == style_img.size(), "we need to import style and content images of the same size"
unloader = transforms.ToPILImage() # reconvert from tensor to PIL image
plt.ion()

