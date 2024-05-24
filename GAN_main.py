
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from HyperTools import *
from Models import *
from advGAN import AdvGAN_Attack
from Gensample import *

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Indian_pines'}



train_loader, test_loader, all_data_loader, y_all = create_data_loader()



use_cuda=True
image_nc = 15
epochs = 100
band_num = 30
BOX_MIN = 0
BOX_MAX = 1
N_PCA, PATCH_SIZE, class_nums = 103, 13, 9

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


pretrained_model = './models/hybrid/data1/13_hybrid1_epoch200_99.88019.pth'
targeted_model = HybridSN(N_PCA, PATCH_SIZE, class_nums).to(device)
# targeted_model = SSRN(N_PCA, PATCH_SIZE, class_nums).to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 16

# MNIST train dataset and dataloader declaration
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(train_loader, epochs, band_num)