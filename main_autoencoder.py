import os
import gin
import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.models as models

from trainer import AutoTrainer
from models import AutoEncoder
from dataset import CustomDatasetFromImages
from dataset import GradedDatasetFromImages

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hacks for Reproducibility
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# from cnn_model import MnistCNNModel
@gin.configurable
def run(batch_size, epochs, val_split, num_workers, print_every,
        trainval_csv_path, test_csv_path, model_type, tasks, lr, weight_decay, 
        momentum, dataset_dir):

    train_dataset = CustomDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)
    # test_dataset = CustomDatasetFromImages(test_csv_path, data_dir = dataset_dir)

    dset_len = len(train_dataset)
    val_size = int(val_split * dset_len)
    test_size = int(0.15 * dset_len)
    train_size = dset_len - val_size - test_size


    train_data, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset,
                                                               [train_size,
                                                                val_size,
                                                                test_size])
    train_loader_small = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2 * batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=False,
                                             drop_last=True,
                                             shuffle=True, 
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=False,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=num_workers)


    if model_type == 'densenet121':
        model = models.densenet121(pretrained=False)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif model_type == 'vgg19':
        model = models.vgg19(pretrained=False)

    model = AutoEncoder(model_type, model = model)
    model = nn.DataParallel(model)

    print(model)

    model = model.to('cuda')

    criterion = nn.MSELoss(reduction='sum')

    # =============================== PRE-TRAIN MODEL ========================
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    trainset_percent = (1 - val_split - 0.15)
    trainer = AutoTrainer(model, optimizer, scheduler, criterion, epochs,
           print_every =  print_every, trainset_split = trainset_percent)
    trainer.train(train_loader, val_loader)
    val_loss = trainer.validate(test_loader)

    with open(trainer.output_log, 'a+') as out:
        print('Test Loss',val_loss, file=out)

if __name__ == "__main__":
    # task_configs =[[0],[1],[2],[0,1], [1,2],[0,2],[0, 1, 2]]
    task_configs = [0.15]
    for i, t in enumerate(task_configs):
        print("Running", (1 -t - 0.15))
    # gin.parse_config_file('config.gin')
        gin.parse_config_file('config_small.gin')
        gin.bind_parameter('run.val_split', t)
        run()
        gin.clear_config()

