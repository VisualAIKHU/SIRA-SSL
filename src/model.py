import numpy as np
import model_base
import torch
import torch.nn.functional as F
from torch import nn
import copy
from PIL import Image
import time


class SIRA_Model(model_base.BaseAttentionModel):
    def __init__(self, args):
        super().__init__(args)
        self.epoch = -1
        self.duplicate_att_module()

    def duplicate_att_module(self):
        self.attmodule_2nd = copy.deepcopy(self.attmodule)
        self.attmodule_3rd = copy.deepcopy(self.attmodule)

    def contrastive_loss(self, img, aud):
        # img is B x 512 x 7 x 7
        # aud is B x 512
        B = img.shape[0]
        if B == 1:
            return 0
        img = self.aud_max_pool(img).squeeze(-1).squeeze(-1)

        # normalize image
        img = F.normalize(img, p=2, dim=1)
        aud = F.normalize(aud, p=2, dim=1)

        logits = torch.mm(img, aud.t())

        positive = torch.diag(logits).view(B, 1)

        # make InfoNCE loss
        loss = torch.exp(positive / 0.07) / \
            torch.sum(torch.exp(logits / 0.07), dim=1)
        loss = -torch.log(loss).mean()

        return loss

    def aud_vis_consistency_loss(self, img, aud):
        # img is B x 512 x 7 x 7
        # aud is B x 512 x 7 x 7
        B = img.shape[0]

        img = img.mean(dim=1)
        aud = aud.mean(dim=1)

        # normalize image with softmax
        flat_img = img.view(B, 49)
        flat_aud = aud.view(B, 49)

        norm_img = F.softmax(flat_img / 0.5, dim=1)
        norm_aud = F.softmax(flat_aud / 0.5, dim=1)

        norm_img = torch.unsqueeze(norm_img, 1)
        norm_aud = torch.unsqueeze(norm_aud, 1)

        def kl_divergence(aud, img):
            # aud, img is B x 49
            B = aud.shape[0]

            if B == 1:
                return 0
            loss = 10 * \
                torch.sum(aud * torch.log((aud+1e-10) / (img + 1e-10)), 1).mean()

            return loss

        loss = kl_divergence(norm_img, norm_aud)

        return loss

    def attention(self, img, aud_vect, flow, att):
        attention, _ = att(img, flow)

        attendedimg = nn.functional.normalize(img + attention, dim=1)

        return self.lvs_loss(attendedimg, aud_vect)

    def update_epoch(self):
        self.epoch += 1

    def forward(self, image, flow, audio, flip=None):
        # Image
        norm_img = self.img_normalize(image)
        img = self.imgnet(norm_img).view(-1, 512, 7, 7)

        # Audio
        aud = self.audnet(audio).view(-1, 512, 9, 9)

        aud_vect = self.aud_max_pool(aud).squeeze(-1).squeeze(-1)
        aud_vect = nn.functional.normalize(aud_vect, dim=1)

        aud = F.interpolate(aud, size=(
            7, 7), mode='bilinear', align_corners=False)

        # return 0, aud.mean(dim=1)

        # min max nomr audn
        norm_aud = (aud - torch.min(aud)) / \
            (torch.max(aud) - torch.min(aud) + 1e-5)

        img_att_by_aud = img * norm_aud

        # Flow
        if self.flowtype == 'cnn':
            flow = self.flownet(flow).view(-1, 512, 7, 7)
        elif self.flowtype == 'maxpool':
            flow = self.flownet(flow)

        loss1, localization1_ = self.attention(
            img, aud_vect, flow, self.attmodule)

        # return loss1, localization1_

        loss2, localization2_ = self.attention(
            img + img_att_by_aud, aud_vect, flow, self.attmodule_2nd)

        # return loss2, localization2_

        localization = localization1_ + localization2_

        localization = localization.unsqueeze(1)
        localization = nn.functional.interpolate(localization, size=(
            224, 224), mode='bilinear', align_corners=False)
        localization = localization.squeeze(1)
        localization = (localization - torch.min(localization)) / \
            (torch.max(localization) - torch.min(localization) + 1e-5)
        localization = torch.where(
            localization > 0.8, torch.ones_like(localization), localization)

        localization = localization.unsqueeze(1)
        localization = torch.cat((localization, localization, localization), 1)

        image_it = image * localization.detach()

        norm_img_it = self.img_normalize(image_it)

        img_it = self.imgnet(norm_img_it).view(-1, 512, 7, 7)

        loss_it, localization_it_ = self.attention(
            img_it, aud_vect, flow, self.attmodule_3rd)

        # return loss_it, localization_it_

        contrastive_loss = self.contrastive_loss(img_it, aud_vect)
        aud_vis_consistency_loss = self.aud_vis_consistency_loss(img_it, aud)

        return loss1 + loss2 + loss_it + contrastive_loss + aud_vis_consistency_loss, localization1_ + localization2_ + localization_it_
