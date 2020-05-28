from skimage import io, transform
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
from my_utils import *
from config import data_transforms, label_dict
import random

data_transforms = data_transforms['val']
display_transform = transforms.Compose([
    transforms.ToPILImage(),
   transforms.Resize((244,244))])


'''
This script is used to create Grad_Cam activation overlays on 
the original images,
adapted from source:  http://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html
currently this script only support resnet models
'''

'''
model can only be resnets for now 
this function saves the activation features as an instance variable
'''
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


def plot_Cam(query_img, label, model_name):
    image = io.imread(query_img)
    tensor = data_transforms(image)
    prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
    model, _ = get_pretrained_models(model_name)
    model.cuda()
    model.eval()
    final_layer = list(model.model.children())[-3]
    activated_features = SaveFeatures(final_layer)
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    pred_label = topk(pred_probabilities, 1).indices[0].item() # get the top predicted class
    pred_label = label_dict[pred_label] # to string
    label = "label: " + label
    pred_label = "predict: " + pred_label
    activated_features.remove()
    weight_softmax_params = list(model.model.fc.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))
    class_idx = topk(pred_probabilities,1)[1].int()
    overlay = getCAM(activated_features.features, weight_softmax, class_idx )
    axes[0].imshow(display_transform(image), cmap='gray')
    axes[0].set_title(label)
    axes[1].imshow(overlay[0], alpha=0.9, cmap='gray')
    axes[2].imshow(display_transform(image), cmap='gray')
    axes[2].imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
    axes[2].set_title(pred_label)
    [axes[i].set_xticks([]) for i in range(3)]
    [axes[i].set_yticks([]) for i in range(3)]
    plt.suptitle("Grad-Cam from model " + model_name, fontsize = 15)
    fig.tight_layout()
    fig.subplots_adjust(top=1.55)
    plt.show()


