#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from skimage import io, transform

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms, utils, models
import torchvision as tv
from torchvision.transforms import v2

import lightning as L
import torchmetrics as tm
import torch.nn.functional as F

from IPython.display import clear_output

import os


# In[2]:


L.seed_everything(42)


# ### Creating the dataset and dataloader from `intel-image-classification`

# In[3]:


# convert PIL image into torch Tensor then does specified transforms from docs: 
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
ds_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[ ]:


def get_dataset(name, **kwargs):
    path = "intel-image-classification/" + name
    ds = tv.datasets.ImageFolder(path, transform=ds_transforms)
    if 'rand_fraction' in kwargs:
        sp_frac = kwargs['rand_fraction']
        if type(sp_frac) is not float or not (0 < sp_frac < 1):
            raise ValueError(f"Invalid `rand_fraction` argument: [{sp_frac}]. Should be a float, s.t. 0.0 < x < 1.0")
        ds, _ = random_split(ds, [sp_frac, 1 - sp_frac])
    return ds

def get_train(aug_name=None, **kwargs):
    if aug_name:
        return get_dataset("seg_train/seg_train_aug/" + aug_name, **kwargs)
    return get_dataset("seg_train/seg_train", **kwargs)

def get_test(**kwargs):
    return get_dataset("seg_test/seg_test", **kwargs)


# In[ ]:


TRAIN_FRACTION = .5


# In[ ]:


train_dataset = get_train(rand_fraction=TRAIN_FRACTION)
test_dataset = get_test()


# In[30]:


len(train_dataset)


# In[31]:


len(test_dataset)


# In[32]:


test_dataset, val_dataset = random_split(test_dataset, [.5, .5])


# ### Creating models & Training

# In[33]:


feature_extractors = {}


# In[34]:


# create resnet50 feature extractor
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.eval()
resnet50_backbone = list(resnet50.children())[:-1]
resnet50_feat_extractor = nn.Sequential(*resnet50_backbone)
feature_extractors['resnet50'] = resnet50_feat_extractor
clear_output()


# In[35]:


class IntelClassifier(L.LightningModule):
    def __init__(self, feature_extractor_name, output_features, num_classes, classifier=None, optimizer=torch.optim.Adam, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        if feature_extractor_name not in feature_extractors:
            raise ValueError(f"`feature_extractor_name` argument is invalid (should be one of {list(feature_extractors.keys())})")
        self.feature_extractor = feature_extractors[feature_extractor_name]
        self.classifier = classifier if classifier else nn.Sequential( # classifier layers after the feature extraction
            nn.Linear(output_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            nn.Linear(512, num_classes)
        )
        self.accuracy = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x).flatten(1)
        return self.classifier(features)

    def _batch_step(self, batch, batch_kind):
        if batch_kind == 'train':
            self.classifier.train()
        else:
            self.classifier.eval()
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
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


# In[36]:


resnet50_model = IntelClassifier('resnet50', 2048, 6, lr=1e-5)


# In[38]:


BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=4)


# In[39]:


# set up loggers
tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir='')
csv_logger = L.pytorch.loggers.CSVLogger(save_dir='')


# In[40]:


trainer = L.Trainer(logger=[tb_logger, csv_logger], callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)], max_epochs=50)


# In[41]:


torch.set_float32_matmul_precision('high')


# In[ ]:


trainer.fit(resnet50_model, train_loader, val_loader)


# In[ ]:


CKPT_PATH = 'resnet50-intel-raw-halftrain.ckpt'


# In[ ]:


trainer.save_checkpoint(CKPT_PATH)


# ### Evaluation

# In[23]:


torch.set_float32_matmul_precision('high')


# In[25]:


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# In[26]:


BATCH_SIZE = 64
test_loader = DataLoader(test_dataset, BATCH_SIZE, num_workers=4)
# train_whole_loader = DataLoader(train_whole_dataset, BATCH_SIZE)
val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=4)


# In[27]:


# y = test_dataset.targets
y = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
# y_train = train_whole_dataset.targets
y_val = np.array([val_dataset[i][1] for i in range(len(val_dataset))])


# In[28]:


def get_test_preds(loaded_model, test_loader):
    trainer = L.Trainer()
    loaded_model.freeze()

    predictions_list = trainer.predict(loaded_model, test_loader) # 30-len list of 32 x 20 tensors
    predictions = torch.vstack(predictions_list).numpy() # 952 x 20
    top_preds = predictions.argmax(axis=1).flatten()

    return top_preds, predictions

def top_preds(all_predictions):
    return (
        np.argsort(all_predictions, axis=1)[:, -5:],
        np.argsort(all_predictions, axis=1)[:, -3:]
    )

def get_topk_accuracy(top_preds, ground_truths):
    ground_truths = np.array(ground_truths)
    #check if ground truth class lies somewhere in the top k
    #check if any of the top 5 predicted classes match the ground truth class
    # print(top_preds.shape)
    ground_truths = ground_truths.reshape(-1, 1)
    matches = np.any(top_preds == ground_truths, axis=1)

    # Count the number of matches
    num_matches = np.sum(matches)
    # print(num_matches)

    # Calculate the percentage of images where at least one of the top 5 predictions matches the ground truth
    percentage_matches = (num_matches / top_preds.shape[0]) * 100
    return percentage_matches

def performance_metrics(predictions, ground_truth, metric_type="Test"):
    accuracy = accuracy_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions, average='weighted')
    precision = precision_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')

    print(f"{metric_type} Accuracy: {accuracy}")
    print(f"{metric_type} Recall: {recall}")
    print(f"{metric_type} Precision: {precision}")
    print(f"{metric_type} F1 Score: {f1}")


# In[29]:


CKPTS = [
    # "resnet50-intel-raw-lr1e-5.ckpt",
    # "resnet50-intel-raw-fulltrain.ckpt",
    # "resnet50-intel-raw-CNET-augmented.ckpt"
    "resnet50-intel-canny-fulltrain.ckpt",
    "resnet50-intel-canny2-fulltrain.ckpt",
]


# In[50]:


import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


# In[49]:


for ckpt in CKPTS:
    print(f"---{ckpt}---")
    loaded_model = IntelClassifier.load_from_checkpoint(checkpoint_path=ckpt)
    # loaded_model = ResnetClassifier(50)
    # checkpoint = torch.load(ckpt)
    # loaded_model.load_state_dict(checkpoint["state_dict"])
    resnet_pred, resnet_all_pred = get_test_preds(loaded_model, val_loader)
    
    resnet_top5, resnet_top3 = top_preds(resnet_all_pred)
    print(f"acc@5 (top 5): {get_topk_accuracy(resnet_top5, y_val)}")
    print(f"acc@3 (top 3): {get_topk_accuracy(resnet_top3, y_val)}")
    
    performance_metrics(resnet_pred, y_val, metric_type="Val")
    print()


# In[47]:


for ckpt in CKPTS:
    print(f"---{ckpt}---")
    loaded_model = IntelClassifier.load_from_checkpoint(checkpoint_path=ckpt)
    # loaded_model = ResnetClassifier(50)
    # checkpoint = torch.load(ckpt)
    # loaded_model.load_state_dict(checkpoint["state_dict"])
    resnet_pred, resnet_all_pred = get_test_preds(loaded_model, test_loader)
    
    resnet_top5, resnet_top3 = top_preds(resnet_all_pred)
    print(f"acc@5 (top 5): {get_topk_accuracy(resnet_top5, y)}")
    print(f"acc@3 (top 3): {get_topk_accuracy(resnet_top3, y)}")
    
    performance_metrics(resnet_pred, y)
    print()


# In[ ]:




