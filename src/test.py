from torch import nn

from src.model.Trainer import Trainer
from src.data.dataset import ChestXrayDataset
import os
from pathlib import Path
from torch.utils.data import DataLoader
from models.models import CXRNet2


conf = {
    "number_of_epochs": 50,
    "val_every": 5,
    "optimizer": "SGD",
    "lr": 0.004,
    "device": "cpu"
}

########################################################
################ GET DATALOADER#########################
########################################################

project_path = Path(__file__).parents[1]
dataset_path = os.path.join(project_path,"data","train")
dataset = ChestXrayDataset(root_dir=dataset_path)
val_set = ChestXrayDataset(root_dir=dataset_path,train=False,train_val_split=0.2)
train_loader = DataLoader(dataset=dataset, batch_size=16)
val_loader = DataLoader(dataset=val_set,batch_size=16)

data,label = dataset[0]
print("Datasets are prepared")
########################################################
################## BUILD MODEL##########################
########################################################

net = CXRNet2(in_channels=1,out_channels=256,in_features=14*14*256,num_classes=2)

loss = nn.MSELoss()

print("Model and loss are prepared")
########################################################
#################### TRAIN #############################
########################################################

print("Starting to train")
trainer = Trainer(model=net, criteria=loss, train_loader=train_loader,val_loader=val_loader,config=conf)
trainer.train_epoch()
print("Done")
