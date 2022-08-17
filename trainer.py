from kitty_dataloader import TinyKitty, image_root
import torch
from torch.utils.data import DataLoader
import os
from torch import optim

from model import ResNetBackBone, Neck, CenterNet, CenterNetHead
from loss import heatMapLoss, whAndOffsetLoss


opj = os.path.join


train_ds = TinyKitty(root=image_root)
train_dl = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn)

# Build the model now its like backbone,neck and Head

# MODEL DEFINITION
backBone = ResNetBackBone()

num_deconv_filters = [256, 128, 64]
num_deconv_kernels = [4, 4, 4]
neck = Neck(backBone.outplanes, num_deconv_filters, num_deconv_kernels)

head = CenterNetHead(in_channel=64, feat_channel=64, num_classes=3)
model = CenterNet(backBone, neck, head)


optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

#img_list, avg_factor, target_result

wh_loss_factor = 1
heat_map_lossFactor = 1
wh_offset_lossFactor = 1

for img_list, avg_factor, target_result in train_dl:
    # print(img_list.shape)
    # print(avg_factor)
    # print(target_result.keys())
    center_heatmap_target = target_result['center_heatmap_target']
    wh_target = target_result['wh_target']
    offset_target = target_result['offset_target']
    whAndOffsetWeight = target_result['offset_target']

    optimizer.zero_grad()
    center_heatmap_pred, wh_pred, offset_pred = model(img_list)

    heatMap_loss = heatMapLoss(
        center_heatmap_pred, center_heatmap_target, avg_factor)
    wh_loss = whAndOffsetLoss(
        wh_pred, wh_target, whAndOffsetWeight, avg_factor)
    wh_offset_loss = whAndOffsetLoss(
        offset_pred, offset_target, whAndOffsetWeight, avg_factor)

    total_loss = heat_map_lossFactor*heatMap_loss + \
        wh_loss_factor*wh_loss+wh_offset_lossFactor*wh_offset_loss
    total_loss.backward
    optimizer.step()
