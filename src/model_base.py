import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
import math
import copy


class BaseAttentionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tau = self.args.tau
        self.flowtype = self.args.flowtype
        self.freeze_vision = self.args.freeze_vision
        self.trimap = self.args.trimap
        self.pretrain_flow = True if self.args.pretrain_flow else False
        self.pretrain_vision = True if self.args.pretrain_vision else False
        self.logit_temperature = self.args.logit_temperature

        self.img_normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        # Vision model
        self.imgnet = resnet18(pretrained=self.pretrain_vision)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()

        # Audio model
        self.audnet = resnet18()
        # Fix first layer channel
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.Identity()
        self.audnet.fc = nn.Identity()

        self.aud_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Flow model
        if self.flowtype == 'cnn':
            self.flownet = resnet18(pretrained=self.pretrain_flow)
            # Fix first layer channel
            self.flownet.conv1 = nn.Conv2d(2, 64, kernel_size=(
                7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.flownet.avgpool = nn.Identity()
            self.flownet.fc = nn.Identity()
            self.attmodule = Self_Attn(512, 512)
        elif self.flowtype == 'maxpool':
            self.flownet = nn.AdaptiveMaxPool2d((7, 7))
            self.attmodule = Self_Attn(512, 2)

        self.m = nn.Sigmoid()
        self.epsilon = self.args.epsilon
        self.epsilon2 = self.args.epsilon - self.args.epsilon_margin

        for m in self.audnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def unfreeze_vision(self, grad):
        for param in self.imgnet.parameters():
            param.requires_grad = grad

    def lvs_loss(self, img, aud):
        B = img.shape[0]  # torch.Size([128, 512, 7, 7]) torch.Size([128, 512])
        self.mask = (1 - 100 * torch.eye(B, B)).to(img.device)

        PosLogits = torch.einsum(
            'ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        Alllogits = torch.einsum(
            'ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])

        Pos = self.m((PosLogits - self.epsilon)/self.tau)

        if self.trimap:
            Pos2 = self.m((PosLogits - self.epsilon2)/self.tau)
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        PosAll = self.m((Alllogits - self.epsilon)/self.tau)

        sim1 = (Pos * PosLogits).view(*
                                      PosLogits.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        sim = ((PosAll * Alllogits).view(*Alllogits.shape[:2], -1).sum(-1) / PosAll.view(
            *PosAll.shape[:2], -1).sum(-1)) * self.mask
        sim2 = (Neg * PosLogits).view(*
                                      PosLogits.shape[:2], -1).sum(-1) / (Neg.view(*Neg.shape[:2], -1).sum(-1))

        logits = torch.cat((sim1, sim, sim2), 1)/self.logit_temperature

        target = torch.zeros((B), dtype=torch.long).to(img.device)

        loss = F.cross_entropy(logits, target)

        return loss, PosLogits.squeeze(1)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, key_in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=key_in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(v).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) / \
            math.sqrt(self.chanel_in//8)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        return out, attention
