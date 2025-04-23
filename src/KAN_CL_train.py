# %%
import os
import time
import math
import pickle
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from torch import device
from torch.utils.data import DataLoader, Subset

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# # DEFINE SETTINGS

# %%
out_path = os.path.join('..', 'results', 'sklearn_digits') # remove or add 'convs'
strategy = "classIL"     # ["taskIL", "classIL"] classIL is harder 
lrs = [1e-3]
decays = [0.8]
epochss = [10]
models = [Efficient_KAN_Fix(strategy, device)]
longer_last_tasks = False
reverse_taks = False

reverse_path = ""
if reverse_taks:
    reverse_path = "reverse_tasks"
longer_last_path = ""
if longer_last_tasks:
    longer_last_path = "longer_last_tasks"

out_path = os.path.join(out_path, strategy, longer_last_path, reverse_path, 'trainings_0423_2')
cfgs = []
for model in models[:1]:
    for lr in lrs:
        for decay in decays:
            for epochs in epochss:
                cfgs.append([model, epochs, lr, decay])

# %% [markdown]
# # Train and test sets

# load sklearn digits
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)
X /= X.max()   # normalize 0→1

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# create 5 class‐IL tasks (2 digits each)
train_loader_tasks = []
for k in range(5):
    cls = {2*k, 2*k+1}
    idxs = [i for i, lbl in enumerate(y_train) if int(lbl) in cls]
    train_loader_tasks.append(
        DataLoader(Subset(train_dataset, idxs), batch_size=50, shuffle=True))

# single test loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if reverse_taks:
    train_loader_tasks.reverse()

# %%
# stats = [0 for i in range(10)]
# for sample in test_dataset:
#     stats[sample[1]] += 1
# print(stats)
# mean = sum(stats)/len(stats)
# variance = sum([((x - mean) ** 2) for x in stats]) / len(stats) 
# res = variance ** 0.5
# print(mean, res)

# %% [markdown]
# ## Trainset visualizer
# The following code prints the images of the 5 domain IL scenarios. This way we can clearly see that for the MNIST dataset each task contains a pair of digits (0-1, 2-3, etc.), while for CIFAR10 each task contains a pair of objects (car-airplane, bird-dog, deer-dog, frog-horse and truck-ship).

# %%
# import numpy as np
# def imshow(img):
#     # img = (img / 2 + 0.5).numpy()
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.axis('off')
#     plt.show()


# def show_images(class_index, num_images=16):
#     dataiter = iter(train_loader_tasks[class_index])
#     images, labels = next(dataiter)
#     imshow(utils.make_grid(images))


# for class_index in range(5):
#     print(f"TASK ID = {class_index}")
#     show_images(class_index)

# %% [markdown]
# # Train and test functions

# %%
def train(model, save_dir, dataloaderobj, optimizer, lr, on_epoch_end, start_epoch=0, epochs=5, isKAN=False):
    criterion = nn.NLLLoss()
    for epoch in range(start_epoch, epochs + start_epoch):
        if not isKAN:
            model.train()
            model.to(device)
        epoch_start = time.time_ns()
        with tqdm(dataloaderobj) as pbar:
            for images, labels in pbar:
                labels = labels.to(device)
                images = images.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step(closure=lambda: loss)
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        epoch_duration = (time.time_ns() - epoch_start) // 1000000
        if on_epoch_end is not None:
            on_epoch_end(model, save_dir, epoch, loss.item(), epoch_duration, lr, isKAN)

# %%
def test(model, isKAN=False):
    if not isKAN:
        model.eval()
    criterion = nn.NLLLoss()
    predictions = []
    ground_truths = []
    val_accuracy = 0
    loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)  #(labels % 2 if model.layers[-1] == 2 else labels).to(device)
            images = images.to(device)
            output = model(images)
            loss = criterion(output, labels)
            predictions.extend(output.argmax(dim=1).to('cpu').numpy())
            ground_truths.extend(labels.to('cpu').numpy())
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    val_accuracy /= len(test_loader)
    print(f"Final Accuracy: {val_accuracy}")
    return loss.item(), ground_truths, predictions

# %%
class EpochStat:
    @staticmethod
    def loadModelStats(dir, name, subdir) -> list['EpochStat']:
        return sorted([pickle.load(open(os.path.join(dir, subdir, file), 'rb')) for file in
                       filter(lambda e: name == '_'.join(e.split('_')[:-1]), os.listdir(os.path.join(dir, subdir)))],
                      key=lambda e: e.epoch)

    def __init__(self, name, save_dir, epoch, train_loss=0, test_loss=0, labels=None, predictions=None, epoch_duration=0, lr=0):
        self.name = name
        self.save_dir = save_dir
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.epoch = epoch
        self.predictions = predictions
        self.labels = labels
        self.epoch_duration = epoch_duration
        self.lr = lr
        self.train_losses = []
        self.train_accuracies = []

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        pickle.dump(self, open(os.path.join(self.save_dir, self.name + '_e' + str(self.epoch) + '.pickle'), 'wb'))

    def get_accuracy(self):
        accuracy = 0
        for label, prediction in zip(self.labels, self.predictions):
            if label == prediction:
                accuracy += 1
        return accuracy / len(self.labels)


def onEpochEnd(model, save_dir, epoch, train_loss, epoch_duration, lr, isKAN):
    test_loss, labels, predictions = test(model, isKAN)
    stat = EpochStat(model.__class__.__name__, save_dir, epoch, train_loss, test_loss, labels, predictions, epoch_duration, lr)
    stat.save()

# %% [markdown]
# # Domain IL - training

# %%
for cfg in cfgs:
    model = cfg[0]
    epochs = cfg[1]
    lr = cfg[2]
    decay_f = cfg[3]
    if decay_f == 1:
        lr_decay = False
    else:
        lr_decay = True
    start_epochs_list = [int(epochs + epochs*i[0]) for i in enumerate(train_loader_tasks)]
    start_epochs_list.insert(0, 0)
    naam = model.__class__.__name__
    isKAN = False
    print("\n\n", naam)
    print(epochs, lr, decay_f, "\n")
    if 'Py_KAN' in naam:
        isKAN = True
    for i, task in enumerate(train_loader_tasks):
        epochs_act = epochs
        if longer_last_tasks and i > 3:
            epochs_act = epochs + epochs

        str_print = f'\t\t\t\tTRAINING ON TASK {i}'
        str_print +=  f' for {epochs_act} epochs' 
        print(str_print)
        # str_epoch = f"ep{epochs}_10fin_"
        str_epoch = f"ep{epochs}"
        str_lr = f"_lr{round(math.log10(lr))}"
        str_decay = '_dec'+ str(decay_f) if lr_decay else ''
        lr_act = lr * decay_f**(i)
        train(model, os.path.join(out_path,f"{str_epoch}{str_lr}{str_decay}", naam), task, optimizer=optim.Adam(model.parameters()),
                lr=lr_act, on_epoch_end=onEpochEnd, start_epoch=start_epochs_list[i], epochs=epochs_act, isKAN=isKAN)
    torch.cuda.empty_cache()

# %%
# PyKAN custom training
# for lr in [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
#     kan = Py_KAN()
    # test(kan)
    # kan.train(lr=lr, train_loader=train_loader_tasks[0])
test(model)

