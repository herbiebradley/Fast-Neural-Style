import os, time

import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from options import Options
from networks import Generator, Vgg16
"""
Required args: --cuda, --data_dir, --save_model_dir, --style_image
"""
def compute_gram_matrix(activations):
    (batch_size, channels, height, width) = activations.size()
    features = activations.view(batch_size, channels, height * width)
    features_trans = features.transpose(1, 2)
    gram = features.bmm(features_trans) / (channels * height * width)
    return gram

if __name__ == "__main__":
    opt = Options().parse(training=True)
    device = torch.device("cuda" if opt.cuda else "cpu")

    generator = Generator(opt).to(device)
    optimizer = Adam(generator.parameters(), opt.learning_rate)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16().to(device)

    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet mean and std
    transform = transforms.Compose([
        transforms.Resize(opt.image_size), # Keeps original aspect ratio
        transforms.CenterCrop(opt.image_size), # Forces images to be square
        transforms.ToTensor(),
        normalise
        ])
    train_dataset = datasets.ImageFolder(opt.data_dir, transform)
    # Use CPU count parallel workers to load the data, and use pinned memory to speed up transfer of data to GPU.
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=os.cpu_count(), pin_memory=True)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        normalise
        ])
    style = style_transform(Image.open(opt.style_image))
    style = style.repeat(opt.batch_size, 1, 1, 1).to(device)

    features_style = vgg(style)
    gram_style_tuple = [compute_gram_matrix(y) for y in features_style]

    for epoch in range(opt.epochs):
        generator.train() # Set training = True, only used by BatchNorm/InstanceNorm/Dropout
        img_count = 0
        for batch_id, (x, _) in enumerate(train_loader): # _ is index of subfolder in data dir
            num_in_batch = len(x)
            img_count += num_in_batch
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True) # Enable asynchronous data transfers to GPU from pinned memory
            y = generator(x)

            features_tuple_y = vgg(y)
            features_tuple_x = vgg(x)

            content_loss = opt.content_weight * mse_loss(features_tuple_y[1], features_tuple_x[1]) # relu2_2

            style_loss = 0.
            for features_y, gram_style in zip(features_tuple_y, gram_style_tuple):
                gram_y = compute_gram_matrix(features_y)
                # Makes sure gram_style is the same size even for last batch
                style_loss += mse_loss(gram_y, gram_style[:num_in_batch, :, :])
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if (batch_id + 1) % opt.log_interval == 0:
                msg = "{}\tEpoch {}:\t[{}/{}]".format(time.ctime(), epoch + 1, img_count, len(train_dataset))
                print(msg)

    generator.eval().cpu() # Set training = False and move generator to cpu to save model
    save_model_filename = "epoch_" + str(opt.epochs) + "_" + str(time.ctime()).replace(' ', '-').replace(':', '') + ".pth"
    save_model_path = os.path.join(opt.save_model_dir, save_model_filename)
    torch.save(generator.state_dict(), save_model_path)
    print("Training Complete")
