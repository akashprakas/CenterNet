# from email.mime import image
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
import xml.etree.ElementTree as ET
from albumentations import Compose, BboxParams, RandomBrightnessContrast, GaussNoise,\
    RGBShift, CLAHE, RandomGamma, HorizontalFlip, RandomResizedCrop, Resize
import cv2
from math import sqrt

opj = os.path.join


image_root = r"C:/Users/akash/Documents/DATA/kitti_tiny"


class Transform(object):
    def __init__(self, box_format='coco', height=512, width=512):
        self.tsfm = Compose([Resize(height=height, width=width),
                             HorizontalFlip(),
                             # RandomResizedCrop(512, 512, scale=(0.75, 1)),
                             RandomBrightnessContrast(0.4, 0.4),
                             GaussNoise(),
                             RGBShift(),
                             CLAHE(),
                             RandomGamma()
                             ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes


def flip(img):
    return img[:, :, ::-1].copy()


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):

    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):

    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def gaussian_radius(det_size, min_overlap):

    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def read_list(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    return lines


class TinyKitty(Dataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    ann_file = 'train.txt'
    img_prefix = 'training/image_2'

    def __init__(self, root, resize=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115),
                 std=(0.28863828, 0.27408164, 0.27809835),
                 ):

        self.down_stride = 4
        self.num_classes = len(self.CLASSES)
        self.cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        self.image_list = read_list(opj(root, self.ann_file))
        self.mode = mode
        self.resize_size = resize
        self.transform = Transform(
            'pascal_voc', height=self.resize_size[0], width=self.resize_size[1])
        self.data_infos = []
        for image_id in self.image_list:
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image_path = opj(root, filename)
            image = Image.open(opj(root, filename))
            image = np.array(image)  # DNT FORGET TO FLIP
            height, width = image.shape[:2]
            data_info = dict(
                filename=image_path, width=width, height=height)

            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            lines = read_list(opj(root, label_prefix, f'{image_id}.txt'))

            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]

            gt_bboxes = []
            gt_labels = []

            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in self.cat2label:
                    gt_labels.append(self.cat2label[bbox_name])
                    gt_bboxes.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long)
            )

            data_info.update(ann=data_anno)

            self.data_infos.append(data_info)

    def __getitem__(self, index):
        data_dict = self.data_infos[index]
        img_h, img_w = data_dict["height"], data_dict["width"]
        image_path = data_dict['filename']
        bboxes = data_dict['ann']['bboxes']
        labels = data_dict['ann']['labels']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = np.array(image)
        if self.mode == "train":
            image, boxes = self.transform(image, bboxes, labels)

        self.imageShape = image.shape
        self.featureMapShape = [image.shape[0]//self.down_stride,
                                image.shape[1]//self.down_stride]
        self.img = transforms.ToTensor()(image)
        self.boxes = torch.Tensor(boxes)
        self.labels = torch.LongTensor(labels)
        return self.img, self.boxes, self.labels

    def __len__(self):
        return len(self.data_infos)

    def collate_fn(self, data):
        # mgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)

        img_list, gt_bboxes, gt_labels = zip(
            *data)

        img_h, img_w = self.imageShape[0], self.imageShape[1]
        bs = len(img_list)
        feat_h, feat_w = self.featureMapShape[0], self.featureMapShape[1]

        width_ratio = float(feat_w/img_w)
        height_ratio = float(feat_h/img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        # the average factor is used , suppose if we have 10 bboxes in one image, then we need to average the loss of those 10 by dividing by 10
        # so the loss from an image with 2 bbox and 10 bbox will be same, otherwise the latter will contribute more.
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)

        img_list = torch.stack(img_list)
        return img_list, avg_factor, target_result


if __name__ == '__main__':
    ds = TinyKitty(root=image_root)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    a = ds[1]
    cat = "nama"
    print(cat)
