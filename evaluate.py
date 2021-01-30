import torch
from torch.utils.data import DataLoader

from models.models import CXRNet2
from src.data.dataset import ChestXrayDataset
from src.model.eval import *
from pathlib import Path
import os

path = os.path.join(Path(__file__).parents[1],"data","test_final")

test_data = ChestXrayDataset(path)
print(test_data.idx_labels)

batch_size = 16
test_loader = DataLoader(test_data, batch_size=batch_size)

in_channels = 1
out_channels = 128
in_features = 14 * 14 * out_channels
num_classes = 4


#m = CXRNet(in_channels, out_channels, in_features, num_classes)
m = CXRNet2(in_channels, out_channels, in_features, num_classes)

model_path = os.path.join(Path(__file__).parents[1],"data","model.pth")
weights = torch.load(model_path)

m.load_state_dict(weights)

criterion = torch.nn.CrossEntropyLoss()

pred, target = eval(m, test_loader, criterion, batch_size, gpu_f=False)
_,_,_,conf_mat = eval_scores(preds=pred,targets=target,path=Path(__file__),file_name="test")
print(conf_mat)
