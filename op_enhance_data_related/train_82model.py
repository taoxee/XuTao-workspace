'''
load data train/val each 50%
train model
use new model to predict test data, return acc, correct_indices, all_preds
get predicts
calculate acc, correct_indices, all_preds

'''
import torch

import sys
sys.path.append("./")


from utils.args_utils import args2str
from utils.constant import DATA_PATH
from models.train_model import *
from utils.data_manger import *
from models import resnet
from utils.model_trainer import *


img_dir = "X:/Python/RESIMAGE50"
data_path='X:/Python/RESIMAGE50'


#train resnet18

current_module = sys.modules[__name__]
if __name__ == '__res18__':
    img_dir = "X:/Python/op_enhance/RESIMAGE50"
    dataset = load_raw_data(DataType.IMAGENET, img_dir)
    first = dataset[0]
    print(type(first[0]), first[1])

current_module = '__res18__'

#run res18
pretrained_path='C:/Users/Tao/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth'
device = f"cuda:0"
train_image_net(device)


'''
#load model
model = resnet.ResNet18()
state_dict = torch.load('./restnet18_8-2_data.pt')
model.load_state_dict(state_dict)
model.eval()
#get acc, correct_indices, all_preds
print(model)



'''
'''
#test
from utils.model_trainer import test

from utils.constant import PretrainedModelPath, DataPath

def test_resnet18():
    seedmodel = ResNet18()
    seedmodel.load_state_dict(torch.load(PretrainedModelPath.IMAGENET.ResNet18))
    data, _ = load_data_set(DATA_IMAGENET,
                         source_data="X:/Python/op_enhance/xutao_workspace",
                         train=False)
    data_loader = DataLoader(dataset='X:/Python/op_enhance/xutao_workspace',
                                  batch_size=32,
                                  num_workers=1)
    premier_acc = test(seedmodel, test_loader=data_loader,device="cuda:0")
    print(premier_acc)



import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL

def test_loader():
    image_transforms = {
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    dataset = 'X:/Python/tao_data'
    test_directory ='X:/Python/tao_data/test'
    data = {'test': datasets.ImageFolder(test_directory, transform=image_transforms['test'])}
    batch_size = 32
    test_data_size = len(data['test'])
    test_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    return test_loader


print("----------------------!!!!!!!!!--------------------------")
print(test_loader)

#get acc, correct_indices, all_preds
aca=test(model, test_loader, verbose=False, device='cuda:0')
print("--------------------------22222!!!!!!!!!-------------------------")
print(aca)
print("----------------333333!!!!!!!!!-------------------")

#test_p=all_preds-correct_indices


#select_wll_samples

from RQ.rq2_labeling import select_wll_samples
select_wll_samples(args=,seed_model=,op_dataset=)
print('wl_samples, torch.tensor(preds).reshape(-1,1), true_lables:',select_wll_samples)


from estimate_wrong_samples.select_wll_samples import *


wl_indices=select_wll_samples_lcr(args='WS',seed_model=ResNet18,test_data_laoder=load_data_set,test_preds=test_p, threshold=1)

'''