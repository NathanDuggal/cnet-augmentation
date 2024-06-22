#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
import torchvision as tv
from torchvision.transforms import v2

import lightning as L
import torchmetrics as tm
import torch.nn.functional as F


# ### Creating the dataset and dataloader from `imagenet-mini`

# In[2]:


labelLookup = pd.read_csv("imagenet-words.txt", delimiter='\t', names=['label'], header=None, index_col=0)['label']


# In[3]:


labelLookup.head()


# In[4]:


# convert PIL image into torch Tensor then does specified transforms from docs: 
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
ds_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[5]:


train_whole_dataset = tv.datasets.ImageFolder("imagenet-mini/train", transform=ds_transforms)
test_dataset = tv.datasets.ImageFolder("imagenet-mini/val", transform=ds_transforms)


# In[6]:


train_whole_dataset


# In[7]:


test_dataset


# In[8]:


train_dataset, val_dataset = random_split(train_whole_dataset, [.9, .1])


# ### Creating Resnet models & Training

# In[9]:


L.seed_everything(42)


# In[10]:


resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }


# In[11]:


class ResnetClassifier(L.LightningModule):
    def __init__(self, variant, lr=1e-2):
        super().__init__()
        if variant not in resnets:
            raise ValueError("`variant` argument is invalid (should be [18, 34, 50, 101, 152])")
        self.resnet_model = resnets[variant](weights=None)
        self.accuracy = tm.classification.Accuracy(task="multiclass", num_classes=1000)
        self.lr = lr

    def forward(self, x):
        return self.resnet_model(x)

    def _batch_step(self, batch, batch_kind):
        if batch_kind == 'train':
            self.resnet_model.train()
        else:
            self.resnet_model.eval()
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        # logging onto tensorboard
        self.log(f"{batch_kind}_loss", loss, prog_bar=True)
        self.log(f"{batch_kind}_acc_f1", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._batch_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._batch_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._batch_step(batch, 'test')

    def predict_step(self, batch, batch_idx):
        self.eval()
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# In[16]:


resnet50_model = ResnetClassifier(50)


# In[17]:


BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=4)


# In[18]:


trainer = L.Trainer(callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")], max_epochs=80)


# In[ ]:


trainer.fit(resnet50_model, train_loader, val_loader)


# In[ ]:


trainer.save_checkpoint("resnet50-imagenetmini-raw.ckpt")

