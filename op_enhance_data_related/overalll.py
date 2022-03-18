import sys
# sys.path.append("../")
from RQ.rq3_overall import *
from utils.data_manger import *
from models.resnet import *
from RQ.rq_utils import *


op_dataset, _ = load_data_set(DATA_IMAGENET,
                     source_data="X:/Python/tao_data/test",
                     train=False)

ori_train_dataset, _ = load_data_set(DATA_IMAGENET,
                     source_data="X:/Python/RESIMAGE50/train",
                     train=True)
ori_test_dataset, _ = load_data_set(DATA_IMAGENET,
                     source_data="X:/Python/RESIMAGE50/test",
                     train=False)

raw_op=load_raw_data(DataType.IMAGENET,"X:/Python/tao_data",False)
raw_oritrain=load_raw_data(DataType.IMAGENET,"X:/Python/RESIMAGE50",True)
raw_oritest=load_raw_data(DataType.IMAGENET,"X:/Python/RESIMAGE50",False)
ori_dataset = load_raw_data(DataType.IMAGENET,'X:/Python/RESIMAGE50',False)


from op_craft import *
mode=op_type
pretrained_path='C:/Users/Tao/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth'
pretrained_model=ResNet18(pretrained_path)

blend_ration=1.0
seed_model=ResNet18()
data_type=DataType.IMAGENET

model_path=DataPath.raw.imagenet
op_type=OperationalDataMaker(DATA_IMAGENET,ori_dataset,blend_ration)
args= read_parameters()
args.data_type=data_type
args.seed_model=seed_model
args.ori_train_dataset=raw_oritrain
args.ori_test_dataset=raw_oritest
args.op_dataset=raw_op
args.select_mode="comb-lr"
args.label_mode="adv"
args.bu=True
args.bound_update=True

def load_pre_mode(data_type):
    data_type = DataType.IMAGENET
    seed_model=ResNet18('C:/Users/Tao/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth')
    # initmodel = torch.load('C:/Users/Tao/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth')
    # seed_model.load_state_dict(initmodel)
    layer_names = ['Linear']
    return seed_model, layer_names

seed_model, last_hidden_layer_name = load_pretrained_model(data_type)
args.seed_model=seed_model
args.last_hidden_layer_name=last_hidden_layer_name

pmap=prepare_model_and_opdata(args)

'''
args = op_enhace(args.data_type, args.op_type, args.select_mode, args.label_mode, args.bu,args.bound_update)
'''