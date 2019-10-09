import sys
sys.path.append('./')
import os
import argparse

import torch
from torchvision import transforms

from tools.imagedb import FaceDataset
from tools.logger import Logger
from models.pnet import PNet
from training.pnet.trainer import PNetTrainer
from training.pnet.train_config import TrainConfig
from checkpoint import CheckPoint
import config

# Get train_config
train_config = TrainConfig()
if not os.path.exists(train_config.save_path):
    os.makedirs(train_config.save_path)

# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = train_config.GPU
use_cuda = train_config.use_cuda and torch.cuda.is_available()
torch.manual_seed(train_config.manualSeed)
torch.cuda.manual_seed(train_config.manualSeed)
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Set dataloader
kwargs = {'num_workers': train_config.nThreads, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_loader = torch.utils.data.DataLoader(
    FaceDataset(train_config.annoPath, transform=transform, is_train=True), batch_size=train_config.batchSize, shuffle=True, **kwargs)

# Set model
model = PNet()
model = model.to(device)

# Set checkpoint
checkpoint = CheckPoint(train_config.save_path)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config.step, gamma=0.1)

# Set trainer
logger = Logger(train_config.save_path)
trainer = PNetTrainer(train_config.lr, train_loader, model, optimizer, scheduler, logger, device)

for epoch in range(1, train_config.nEpochs + 1):
    cls_loss_, box_offset_loss, total_loss, accuracy = trainer.train(epoch)
    checkpoint.save_model(model, index=epoch, tag=str(config.NUM_LANDMARKS) + '_landmarks')

