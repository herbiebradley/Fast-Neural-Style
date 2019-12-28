import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from options import Options
from networks import Generator
"""
Required args: --cuda, --content_image, --output_image, --model
"""
if __name__ == "__main__":
    opt = Options().parse(training=False)
    device = torch.device("cuda" if opt.cuda else "cpu")

    content_image = Image.open(opt.content_image)
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    content_transform = transforms.Compose([transforms.ToTensor(), normalise])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = Generator(opt)
        state_dict = torch.load(opt.model) # Investigate state_dict
        style_model.load_state_dict(torch.load(opt.model))

        style_model.to(device)
        output = style_model(content_image).cpu()
    # save image
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1.0/0.229, 1.0/0.224, 1.0/0.255])
    result = inv_normalize(output[0].clone()) # Reverse normalisation
    result = result.mul(255).clamp(0, 255).numpy() # Scale up to [0, 255] and convert to numpy.
    result = result.transpose(1, 2, 0).astype("uint8") # Convert to H x W x C integer array.
    result = Image.fromarray(result)
    result.save(opt.output_image)