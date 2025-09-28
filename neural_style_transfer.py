import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_loader(img_name, imsize):
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
    img = Image.open(img_name)
    img_tensor: torch.Tensor = loader(img)  # type: ignore
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(device, torch.float)


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def gram_matrix(input):
    a, b, c, d = input.size() # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t()) # compute the gram product

    return G.div(a*b*c*d) # normalize values of gram matrix by dividing by number of element in each feature maps


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach() # we don't want to backprop through this

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
    
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

    

# Load VGG19 model with pre-trained weights
cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)


        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)


    # trim off layers after last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses
    


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)


    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = torch.tensor(0.0, device=device, requires_grad=True)
            content_score = torch.tensor(0.0, device=device, requires_grad=True)

            for style_layer in style_losses:
                style_score = style_score + (1/5) * style_layer.loss

            for content_layer in content_losses:
                content_score = content_score + content_layer.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return (style_score + content_score).detach()
        
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

