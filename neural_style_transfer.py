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


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach() # we don't want to backprop through this

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    

# for style loss
def gram_matrix(input):
    a, b, c, d = input.size() # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t()) # compute the gram product

    return G.div(a*b*c*d) # normalize values of gram matrix by dividing by number of element in each feature maps
    

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
    
# helps to normalize input image so that it can be easily fit into VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

    
# Importing VGG 19 model like in the paper
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



