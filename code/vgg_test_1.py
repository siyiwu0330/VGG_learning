# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import datasets ,models, transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import tqdm

# Root directory for dataset
raw_root = "/content/drive/MyDrive/code/datasets/cat_classification/cat_classification/raw"
train_root = "/content/drive/MyDrive/code/datasets/cat_classification/cat_classification/train"
val_root = "/content/drive/MyDrive/code/datasets/cat_classification/cat_classification/val"
test_root = "/content/drive/MyDrive/code/datasets/cat_classification/cat_classification/test"

# Batch size during training
batch_size = 8

# Number of workers for dataloader
workers = 0

# size using a transformer.
image_size = 500

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.000001

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Create the dataset
raw_transforms = datasets.ImageFolder( root=raw_root,
                                  transform=transforms.Compose([
                                                      transforms.Resize(image_size),
                                                      transforms.CenterCrop(image_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                      ]))
train_transforms = datasets.ImageFolder( root=train_root,
                                  transform=transforms.Compose([
                                                      transforms.Resize(image_size),
                                                      transforms.CenterCrop(image_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                      ]))
val_transforms = datasets.ImageFolder( root=val_root,
                                  transform=transforms.Compose([
                                                      transforms.Resize(image_size),
                                                      transforms.CenterCrop(image_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                      ]))
test_transforms = datasets.ImageFolder( root=test_root,
                                  transform=transforms.Compose([
                                                      transforms.Resize(image_size),
                                                      transforms.CenterCrop(image_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                      ]))

# Create the dataloader
raw_loader = torch.utils.data.DataLoader(raw_transforms, batch_size = batch_size, shuffle=True, num_workers=workers)
train_loader = torch.utils.data.DataLoader(train_transforms, batch_size = batch_size, shuffle=True, num_workers=workers)
val_loader = torch.utils.data.DataLoader(val_transforms, batch_size = batch_size, shuffle=True, num_workers=workers)
test_loader = torch.utils.data.DataLoader(val_transforms, num_workers=workers)

# classes
classes = ['0','1','2','3','4','5','6','7','8','9','10','11']


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some raw images
real_batch = next(iter(raw_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Raw Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

print(train_transforms.class_to_idx)
print(val_transforms.class_to_idx)

vgg_19 = models.vgg19_bn(pretrained=True)
vgg_19

for param in vgg_19.parameters():
    param.requires_grad = True

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(0.3)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('relu2', nn.ReLU()), 
                          ('dropout2',nn.Dropout(0.3)),
                          ('fc3', nn.Linear(512, 12)),
                          ('softmax', nn.LogSoftmax(dim=1))
                          ]))

vgg_19.classifier = classifier

vgg_19.to(device)

print(vgg_19)

# Initialize CrossEntropyLoss function
criterion = nn.CrossEntropyLoss()

# Setup Adam optimizers ResNet
optimizer = optim.Adam(vgg_19.parameters(), lr=lr)

# track change in validation loss
valid_loss_min = np.Inf

training_losses = []
valid_losses = []

for epoch in range(1, num_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    counter = 1
    
    #######################
    # train the model #
    #######################
    vgg_19.train()
    counter = 1
    for data, target in tqdm.notebook.tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg_19(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

        # Output training stats
        if counter % 10 == 0:
          print('[%d/%d] Training Loss: %.4f\t' % (counter, len(train_loader), train_loss/(counter * batch_size)))
        counter += 1
        
    ##########################    
    # validate the model #
    ##########################
    vgg_19.eval()
    counter = 1
    for data, target in tqdm.notebook.tqdm(val_loader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg_19(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)

        # Output valid stats
        if counter % 5 == 0:
          print('[%d/%d] Valid Loss: %.4f\t' % (counter, len(val_loader), valid_loss/(counter * batch_size)))
        counter += 1
    
    # calculate average losses
    # train_losses.append(train_loss/len(train_loader.dataset))
    # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(val_loader.dataset)

    training_losses.append(train_loss)
    valid_losses.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
        epoch, num_epochs, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(vgg_19.state_dict(), 'model_vgg19_1.pth')
        valid_loss_min = valid_loss

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(training_losses,label="TLoss")
plt.plot(valid_losses,label="VLoss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

batch_size = 4
valid_loader = torch.utils.data.DataLoader(val_transforms, batch_size=batch_size,  num_workers = workers, shuffle=True)
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(12))
class_total = list(0. for i in range(12))

vgg_19.eval()
# iterate over valid data
for data, target in valid_loader:
    # move tensors to GPU if CUDA is available
    data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg_19(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(valid_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(12):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
