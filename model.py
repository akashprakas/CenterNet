import torch
from torchvision import models
import torch.nn as nn


class ResNetBackBone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # we will use a resnet 18 by default
        # we will remove the last two blocks, the adaptive pool and fc layer
        resnet = models.resnet18(pretrained=pretrained)
        blockList = list(resnet.children())
        self.featureMap = nn.Sequential(*blockList[:-2])
        self.outplanes = 512

    def forward(self, x):
        out = self.featureMap(x)
        return out


class Neck(nn.Module):
    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels):
        super().__init__()
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

    def ConvModule(self, in_channels, feat_channels, kernel_size, stride, padding):
        convLayers = [nn.Conv2d(in_channels, feat_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=False),
                      nn.BatchNorm2d(feat_channels),
                      nn.ReLU(inplace=True)]

        return convLayers

    def DeconvModule(self, in_channels, feat_channels,  kernel_size, stride, padding):
        deConvLayers = [nn.ConvTranspose2d(
            in_channels, feat_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)]

        return deConvLayers

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = self.ConvModule(
                self.in_channel,
                feat_channel,
                3, stride=1,
                padding=1
            )
            layers.extend(conv_module)
            upsample_module = self.DeconvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1
            )
            layers.extend(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.deconv_layers(x)
        return out


class CenterNetHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes):

        super().__init__()
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch

        # def init_weights(self):
        # """Initialize weights of the head."""
        # bias_init = bias_init_with_prob(0.1)
        # self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        #     for m in head.modules():
        #         if isinstance(m, nn.Conv2d):
        #             normal_init(m, std=0.001)

        pass

    def forward(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """

        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred


class CenterNet(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)
        return out






if __name__ == '__main__':

    # Tests
    testInput = torch.randn((2, 3, 512, 512))
    backBone = ResNetBackBone()
    backBoneOut = backBone(testInput)
    # output should be torch.Size([2, 512, 16, 16])
    print("Backbone output ", backBoneOut.shape)

    # deconv layer
    num_deconv_filters = [256, 128, 64]
    num_deconv_kernels = [4, 4, 4]
    neck = Neck(backBone.outplanes, num_deconv_filters, num_deconv_kernels)
    neckOut = neck(backBoneOut)
    print("Ouput of the neck is ", neckOut.shape)

    # Head output

    head = CenterNetHead(in_channel=64, feat_channel=64, num_classes=3)
    heatmap, wh, whOffset = head(neckOut)

    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("whOffset shape", whOffset.shape)

    print("==================================================")
    CT = CenterNet(backBone, neck, head)
    heatmap, wh, whOffset = CT(testInput)

    print("Heatmap shape ", heatmap.shape)
    print("wh shape", wh.shape)
    print("whOffset shape", whOffset.shape)

    name = "akash"
    print(name)
