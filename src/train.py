import copy
import os
import time
from matplotlib import cm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import SIRA_DataModule
from model import SIRA_Model
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from sklearn import metrics
import xml.etree.ElementTree as ET
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn


def convert_to_vggss(filename):
    return filename.replace(".mp4", "")


def testset_gt(args, name):
    if 'flickr' in args.testset:
        gt = ET.parse(args.gt_path + f'{name}.xml').getroot()
        if len(gt) == 0:
            print('empty gt')
        gt_map = np.zeros([224, 224])
        bboxs = []
        for child in gt:
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index, ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[item_[1]:item_[3], item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map > 1] = 1

    elif 'vggss' in args.testset:
        if not hasattr(args, 'gt_all'):
            args.gt_all = {}
            with open('metadata/vggss.json') as json_file:
                annotations = json.load(json_file)
            for annotation in annotations:
                args.gt_all[convert_to_vggss(
                    annotation['file'])] = annotation['bbox']
        gt = args.gt_all[name]
        gt_map = np.zeros([224, 224])
        for item_ in gt:
            item_ = list(map(lambda x: int(224 * max(x, 0)), item_))
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[ymin:ymax, xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map > 0] = 1
    return gt_map


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='unspecified',
                        type=str, help='experiment name')
    parser.add_argument('--trainset', default='vggss',
                        type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss',
                        type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='',
                        type=str, help='Root directory path of data')
    parser.add_argument('--test_data_path', default='',
                        type=str, help='Root directory path of data')
    parser.add_argument('--image_size', default=224,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--gt_path', default='', type=str)
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon_margin', default=0.25, type=float)
    parser.add_argument('--logit_temperature', default=0.07, type=float)
    parser.add_argument('--trimap', default=1, type=int)
    parser.add_argument('--pretrain_flow', default=1, type=int)
    parser.add_argument('--pretrain_vision', default=1, type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--flow', type=int, default=1)
    parser.add_argument('--flowtype', type=str, default='cnn')
    parser.add_argument('--freeze_vision', type=int, default=1)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--ckpt', default='', type=str,
                        help='filepath of weights')
    parser.add_argument('--description', default='',
                        type=str, help='description of experiment')

    return parser.parse_args()


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


class Evaluator:
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) +
                                          np.sum(infer_map * (gtmap == 0)))
        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap == 0)))

    def final(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []


class SIRA_Module(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = SIRA_Model(self.args)

        self.evaluator = Evaluator()

        self.iou = []

        if args.freeze_vision:
            print("FREEZING IMAGE ENCODER")
            self.model.unfreeze_vision(False)

        # gt for vggss
        if args.testset == 'vggss':
            args.gt_all = {}
            with open('metadata/vggss.json') as json_file:
                annotations = json.load(json_file)
            for annotation in annotations:
                args.gt_all[convert_to_vggss(
                    annotation['file'])] = annotation['bbox']

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        image, flow, spec, file_id, flip = batch

        loss, attention = self.model(
            image.float(), flow.float(), spec.float(), flip)

        self.log("train/loss", loss.item(), prog_bar=False,
                 on_epoch=True, batch_size=self.args.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        image, flow, spec, file_id, _ = batch

        loss, localization = self.model(
            image.float(), flow.float(), spec.float())

        localization = localization.squeeze(0).cpu().numpy()
        localization_map = cv2.resize(localization, dsize=(
            224, 224), interpolation=cv2.INTER_CUBIC)

        for i in range(1):
            sample_localization = normalize_img(localization_map)

            sample_gt = testset_gt(self.args, file_id[i])

            thr = np.sort(sample_localization.flatten())[
                int(sample_localization.shape[0] * sample_localization.shape[1] / 2)]

            sample_localization[sample_localization > thr] = 1
            sample_localization[sample_localization < 1] = 0
            self.evaluator = Evaluator()
            ciou_lvs, inter, union = self.evaluator.cal_CIOU(
                sample_localization, sample_gt, 0.5)

            self.iou.append(ciou_lvs)

        return {"loss": loss, "localization": localization}

    def test_step(self, batch, batch_idx):
        image, flow, spec, file_id, _ = batch

        loss, localization = self.model(
            image.float(), flow.float(), spec.float())

        localization = localization.squeeze(0).cpu().numpy()
        localization_map = cv2.resize(localization, dsize=(
            224, 224), interpolation=cv2.INTER_CUBIC)

        for i in range(1):
            sample_localization = normalize_img(localization_map)

            sample_gt = testset_gt(self.args, file_id[i])

            thr = np.sort(sample_localization.flatten())[
                int(sample_localization.shape[0] * sample_localization.shape[1] / 2)]

            sample_localization[sample_localization > thr] = 1
            sample_localization[sample_localization < 1] = 0

            self.evaluator = Evaluator()
            ciou_lvs, inter, union = self.evaluator.cal_CIOU(
                sample_localization, sample_gt, 0.5)

            self.iou.append(ciou_lvs)

        return {"loss": loss, "localization": localization}

    def validation_epoch_end(self, outputs):

        results = []
        for i in range(21):
            result = np.sum(np.array(self.iou) >= 0.05 * i)
            result = result / len(self.iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = metrics.auc(x, results)
        thresholds = np.arange(0.5, 1.0, 0.05)
        ciou_values = []
        ciou_0_5 = np.sum(np.array(self.iou) >= 0.5)/len(self.iou)
        for threshold in thresholds:
            ciou = np.sum(np.array(self.iou) >= threshold) / len(self.iou)
            ciou_values.append(ciou)
        mciou = np.mean(ciou_values)
        self.iou = []

        self.iou = []

        self.log("val/mcIoU", mciou, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val/cIoU", ciou_0_5, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val/AUC", auc_, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)

        self.model.update_epoch()

        print("epoch: ", self.model.epoch)

        self.evaluator = Evaluator()

    def test_epoch_end(self, outputs):

        results = []
        for i in range(21):
            result = np.sum(np.array(self.iou) >= 0.05 * i)
            result = result / len(self.iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = metrics.auc(x, results)
        thresholds = np.arange(0.5, 1.0, 0.05)
        ciou_values = []
        ciou_0_5 = np.sum(np.array(self.iou) >= 0.5)/len(self.iou)
        for threshold in thresholds:
            ciou = np.sum(np.array(self.iou) >= threshold) / len(self.iou)
            ciou_values.append(ciou)
        mciou = np.mean(ciou_values)
        self.iou = []

        self.log("test/cIoU", ciou_0_5, prog_bar=True,
                 on_epoch=True, batch_size=1)
        self.log("test/mcIoU", mciou, prog_bar=True,
                 on_epoch=True, batch_size=1)
        self.log("test/AUC", auc_, prog_bar=True, on_epoch=True, batch_size=1)

        self.evaluator = Evaluator()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer


if __name__ == "__main__":
    args = get_arguments()

    module = SIRA_Module(args)

    datamodule = SIRA_DataModule(args)

    checkpointer = ModelCheckpoint(dirpath=f"logs/{args.name}/",
                                   filename='best',
                                   monitor="val/cIoU",
                                   save_last=True,
                                   save_weights_only=False,
                                   mode='max')

    logger = TensorBoardLogger("logs", name=args.name)

    trainer = Trainer(accelerator='gpu',
                      devices="auto",
                      strategy="ddp",
                      max_epochs=args.epochs,
                      num_sanity_val_steps=400,
                      callbacks=[checkpointer],
                      logger=logger)

    trainer.fit(module, datamodule)
