
import torch
import numpy as np
import time
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import sys

sys.path.append('../')
from utils.data_manger import *
from utils.constant import *

device = torch.device('cuda:0')

#import model
import sys
sys.path.append("./")
from models.resnet import ResNet18
from collections import OrderedDict
from utils.commons import*

from models import *
from utils.data_manger import *
import os

data, _ = load_data_set(DATA_IMAGENET,
                     source_data="X:/Python/tao_data/test",
                     train=False)
data_loader = DataLoader(dataset=data,
                                batch_size=64,
                                num_workers=1)

test_loader=data_loader
device="cuda:0"

path = "X:/Python/op_enhance/restnet18_8-3_data.pt"
ini_model = ResNet18()
ini_model.to("cuda:0")
ini_model.eval()

#model=res18('X:/Python/op_enhance/restnet18_8-2_data.pt')

#get prediction way
def test(model, test_loader, verbose=True, device='cpu'):
    test_loss = 0
    correct = 0
    progress = 0
    batch_size = test_loader.batch_size
    print ("--------------------------------BATCHSIZE",batch_size)
    data_size = len(test_loader.dataset)
    print("--------------------------------DADADADADATA SIZE",data_size)
    time_count = []
    all_preds = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            #print(sample)
            data, target = data.to(device), target.to(device)
            #data, target = sample['data'].to(device), sample['target'].to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            start = time.process_time()
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy().tolist())
            time_count.append(time.process_time() - start)
            progress += 1
            if verbose:
                sys.stdout.write('\r progress:{:.2f}%'.format(
                    (1. * batch_size * progress * 100) / data_size))
    test_loss /= len(test_loader.dataset)
    acc = 1. * correct / len(test_loader.dataset)
    print("----------------------------------ACC:",acc)
    true_labels = test_loader.dataset.targets
    all_preds = np.array([e[0] for e in all_preds])
    print ("---------------------------------ALL Preds:",all_preds)
    correct_indices = np.where(all_preds == true_labels)[0]
    print ('---------------------------------Correct_Indices: ',correct_indices)
    assert len(correct_indices) == correct, f"{len(correct_indices)}, {correct}"
    if verbose:
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%) time:{:.6f}\n'
                .format(test_loss, correct, len(test_loader.dataset), 100 * acc,
                        np.average(time_count)))
    torch.cuda.empty_cache()
    return acc, correct_indices, all_preds

device= torch.device('cuda:0')
ac,ci,ap=test(ini_model,test_loader,True, device)
print(ac)
print(ci.shape,ap.shape)


cii=np.array(ci)
app=np.array(ap)
cp=app-cii
print(cp)
