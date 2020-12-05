import os
import random

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from settings import DEFAULT_SETTING
from utils.utils import batch_warp_points, normPts, denormPts, inv_warp_image_batch
from torch.nn.functional import interpolate

class UnSuperPoint(nn.Module):
    def __init__(self, config=None):
        super(UnSuperPoint, self).__init__()
        self.config = config
        if not config:
            self.config = DEFAULT_SETTING
        self.usp = self.config['model']['usp_loss']['alpha_usp']
        self.position_weight = self.config['model']['usp_loss']['alpha_position']
        self.score_weight = self.config['model']['usp_loss']['alpha_score']
        self.uni_xy = self.config['model']['unixy_loss']['alpha_unixy']
        self.desc = self.config['model']['desc_loss']['alpha_desc']
        self.d = self.config['model']['desc_loss']['lambda_d']
        self.m_p = self.config['model']['desc_loss']['margin_positive']
        self.m_n = self.config['model']['desc_loss']['margin_negative']
        self.decorr = self.config['model']['decorr_loss']['alpha_decorr']
        self.correspond = self.config['model']['correspondence_threshold']
        self.conf_thresh = self.config['model']['detection_threshold']
        self.nn_thresh = self.config['model']['nn_thresh']

        self.border_remove = 4  # Remove points this close to the border.
        self.downsample = 8
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32,32,3,1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.score = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,1,3,1,padding=1),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256,2,3,1,padding=1),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(256, 256,3,1,padding=1)
        )

    def forward(self, x):
        h,w = x.shape[-2:]
        self.h = h
        self.w = w
        output = self.cnn(x)
        s = self.score(output)
        p = self.position(output)
        d = self.descriptor(output)
        desc = self.interpolate(p, d, h, w)
        return s,p,desc

    def interpolate(self, p, d, h, w):
        # b, c, h, w
        # h, w = p.shape[2:]
        samp_pts = self.get_bath_position(p)
        samp_pts[:, 0, :, :] = (samp_pts[:, 0, :, :] / (float(self.w)/2.)) - 1.
        samp_pts[:, 1, :, :] = (samp_pts[:, 1, :, :] / (float(self.h)/2.)) - 1.
        samp_pts = samp_pts.permute(0,2,3,1)
        desc = torch.nn.functional.grid_sample(d, samp_pts)
        return desc

    def tb_add_loss(self, loss, task='train'):
        for k in loss:
            self.writer.add_scalar('{}/{}'.format(task, k), loss[k].item(), self.step)

    @torch.no_grad()
    def tb_add_hist(self, name, ts, bin=100, task='train'):
        f = plt.figure()
        plt.hist(ts.reshape(-1).cpu().numpy(), bin, density=True)
        plt.close()
        self.writer.add_figure('{}/{}'.format(task, name), f, self.step)

    def train_val_step(self, img0, img1, mat, task='train'):
        self.task = task
        #with torch.autograd.detect_anomaly():
        img0 = img0.to(self.dev)
        img1 = img1.to(self.dev)
        mat = mat.squeeze()
        mat = mat.to(self.dev)
        self.optimizer.zero_grad()
        
        if task == 'train':
            s1,p1,d1 = self.forward(img0)
            s2,p2,d2 = self.forward(img1)

        else:
            with torch.no_grad():
                s1,p1,d1 = self.forward(img0)
                s2,p2,d2 = self.forward(img1)

        lossdict = self.loss(s1,p1,d1,s2,p2,d2,mat)
        self.tb_add_loss(lossdict, task)

        if task == 'train' and self.step % self.config['tensorboard_interval'] == 0:
            self.tb_add_hist('train/left/x_relative', p1[0])
            self.tb_add_hist('train/left/y_relative', p1[1])
            self.tb_add_hist('train/left/score', s1)
            self.tb_add_hist('train/right/x_relative', p2[0])
            self.tb_add_hist('train/right/y_relative', p2[1])
            self.tb_add_hist('train/right/score', s2)

        if task == 'valid':
            self.tb_add_hist('valid/left/x_relative', p1[0])
            self.tb_add_hist('valid/left/y_relative', p1[1])
            self.tb_add_hist('valid/left/score', s1)
            self.tb_add_hist('valid/right/x_relative', p2[0])
            self.tb_add_hist('valid/right/y_relative', p2[1])
            self.tb_add_hist('valid/right/score', s2)
            # TODO: Validation operations

        if task == 'train':
            lossdict['loss'].backward()
            self.optimizer.step()

        return lossdict['loss'].item()
    
    def train_val_step_tri(self, img0, img1, img, mat1, mat2, mask1, mask2, task='train'):
        self.task = task
        #with torch.autograd.detect_anomaly():
        img0 = img0.to(self.dev)
        img1 = img1.to(self.dev)
        img = img.to(self.dev)
        mat1 = mat1.squeeze()
        mat1 = mat1.to(self.dev)
        mat2 = mat2.squeeze()
        mat2 = mat2.to(self.dev)
        self.optimizer.zero_grad()
        
        if task == 'train':
            s1,p1,d1 = self.forward(img0)
            s2,p2,d2 = self.forward(img1)
            s, p, d = self.forward(img)

        else:
            with torch.no_grad():
                s1,p1,d1 = self.forward(img0)
                s2,p2,d2 = self.forward(img1)
                s, p, d = self.forward(img)

        lossdict = self.triloss(s1,p1,d1,s2,p2,d2,s,p,d,mat1,mat2,mask1,mask2)
        self.tb_add_loss(lossdict, task)

        if task == 'train' and self.step % self.config['tensorboard_interval'] == 0:
            self.tb_add_hist('train/A/x_relative', p1[0])
            self.tb_add_hist('train/A/y_relative', p1[1])
            self.tb_add_hist('train/A/score', s1)
            self.tb_add_hist('train/B/x_relative', p2[0])
            self.tb_add_hist('train/B/y_relative', p2[1])
            self.tb_add_hist('train/B/score', s2)
            self.tb_add_hist('train/C/x_relative', p[0])
            self.tb_add_hist('train/C/y_relative', p[1])
            self.tb_add_hist('train/C/score', s)

        if task == 'valid':
            self.tb_add_hist('valid/A/x_relative', p1[0])
            self.tb_add_hist('valid/A/y_relative', p1[1])
            self.tb_add_hist('valid/A/score', s1)
            self.tb_add_hist('valid/B/x_relative', p2[0])
            self.tb_add_hist('valid/B/y_relative', p2[1])
            self.tb_add_hist('valid/B/score', s2)
            self.tb_add_hist('valid/C/x_relative', p[0])
            self.tb_add_hist('valid/C/y_relative', p[1])
            self.tb_add_hist('valid/C/score', s)
            # TODO: Validation operations

        if task == 'train':
            lossdict['loss'].backward()
            self.optimizer.step()

        return lossdict['loss'].item()
        
    # Loss functions for original UnsuperPoint
    def loss(self, bath_As, bath_Ap, bath_Ad, 
        bath_Bs, bath_Bp, bath_Bd, mat):
        usp = 0; unixy = 0; desc = 0; decorr = 0
        usp, unixy, desc, decorr = self.UnSuperPointLoss(bath_As, bath_Ap, bath_Ad, bath_Bs, bath_Bp, bath_Bd, mat)

        if torch.isnan(usp).any():
            loss = unixy + desc + decorr
            lossdict = {
                "loss": loss,
                "uni_xy_loss": unixy,
                "descriptor_loss": desc,
                "decorrelation_loss": decorr
            }
            
            return lossdict

        loss = usp + unixy + desc + decorr
        lossdict = {
            "loss": loss,
            "usp_loss": usp,
            "uni_xy_loss": unixy,
            "descriptor_loss": desc,
            "decorrelation_loss": decorr
        }
        
        return lossdict

    def UnSuperPointLoss(self, As, Ap, Ad, Bs, Bp, Bd, mat):
        position_A = self.get_position(Ap, mat=mat)
        position_B = self.get_position(Bp)
        G = self.getG(position_A, position_B)

        Usploss = self.usploss(As, Bs, G)
        Uni_xyloss = self.uni_xyloss(Ap, Bp)

        Descloss = self.descloss(Ad, Bd, G)
        Decorrloss = self.decorrloss(Ad, Bd)
        return self.usp * Usploss, self.uni_xy * Uni_xyloss,\
            self.desc * Descloss, self.decorr * Decorrloss

    def get_position(self, Pmap, mat=None):
        x = 0
        y = 1
        res = torch.zeros_like(Pmap)
        # print(Pmap.shape,res.shape)
        for i in range(Pmap.shape[3]):
            res[:,x,:,i] = (i + Pmap[:,x,:,i]) * self.downsample
        for i in range(Pmap.shape[2]):
            res[:,y,i,:] = (i + Pmap[:,y,i,:]) * self.downsample 
        if mat is not None:
            # print(mat.shape)
            shape = torch.tensor([Pmap.shape[3], Pmap.shape[2]]).to(self.dev) * self.downsample
            B = Pmap.shape[0]
            Hc, Wc = Pmap.shape[2:]
            res = normPts(res.permute(0, 2, 3, 1).reshape((B, -1, 2)), shape)
            # r = torch.stack((res[:,1], res[:,0]), dim=1) # (y, x) to (x, y)
            r = batch_warp_points(res, mat, self.dev)
            # r = torch.stack((r[:,1], r[:,0]), dim=1)  # (x, y) to (y, x)
            r = denormPts(r, shape).reshape(B, Hc, Wc, 2).permute(0, 3, 1, 2)
            return r
        else:
            return res

    def usploss(self, As, Bs, G):
        A2BId, Id = self.get_point_pair(G)
        # print(A2BId.shape, Id.shape)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        B = G.shape[0]
        reshape_As = As.reshape(B, -1)
        reshape_Bs = Bs.reshape(B, -1)
        positionK_loss = scoreK_loss = uspK_loss = 0

        for i in range(B):
            d_k = G[i][Id[i], A2BId[i][Id[i]]]
            reshape_As_k = reshape_As[i][Id[i]]
            reshape_Bs_k = reshape_Bs[i][A2BId[i][Id[i]]]
            positionK_loss += torch.mean(d_k)
            scoreK_loss += torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
            uspK_loss += self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)

        return (self.position_weight * positionK_loss + 
            self.score_weight * scoreK_loss + uspK_loss)/B

    def get_bath_position(self, Pamp):
        x = 0
        y = 1
        res = torch.zeros_like(Pamp)
        for i in range(Pamp.shape[3]):
            res[:,x,:,i] = (i + Pamp[:,x,:,i]) * self.downsample
        for i in range(Pamp.shape[2]):
            res[:,y,i,:] = (i + Pamp[:,y,i,:]) * self.downsample
        return res

    def getG(self, PA, PB):
        b, c = PA.shape[0:2]
        # reshape_PA shape = b, 2, m -> b, m, 2
        reshape_PA = PA.reshape((b, c, -1)).permute(0, 2, 1)
        # reshape_PB shape = b, 2, m -> b, m, 2
        reshape_PB = PB.reshape((b, c, -1)).permute(0, 2, 1)
        # x shape b,m,m <- b, (m,1 - 1,m) 
        x = torch.unsqueeze(reshape_PA[:,:,0],2) - torch.unsqueeze(reshape_PB[:,:,0],1)
        # y shape b,m,m <- b, (m,1 - 1,m)
        y = torch.unsqueeze(reshape_PA[:,:,1],2) - torch.unsqueeze(reshape_PB[:,:,1],1)

        G = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))

        return G

    def get_point_pair(self, G):
        A2B_min_Id = torch.argmin(G, dim=2)
        B, M = A2B_min_Id.shape[0:2]
        Id = torch.nn.functional.grid_sample(G.unsqueeze(1), 
            torch.stack([torch.arange(M).reshape(1, M).repeat(B, 1).to(self.dev), A2B_min_Id],
                dim=2).unsqueeze(2).float())
        Id = Id.squeeze() <= self.correspond
        return A2B_min_Id, Id

    def get_uspK_loss(self, d_k, reshape_As_k, reshape_Bs_k):
        sk_ = (reshape_As_k + reshape_Bs_k) / 2
        d_ = torch.mean(d_k)
        return torch.mean(sk_ * (d_k - d_))

    def uni_xyloss(self, Ap, Bp):
        b, c = Ap.shape[0:2]
        reshape_PA = Ap.reshape((b,c,-1)).transpose(2,1)
        reshape_PB = Bp.reshape((b,c,-1)).transpose(2,1)
        loss = 0
        for i in range(2):
            loss += self.get_uni_xy(reshape_PA[:,:,i], b)
            loss += self.get_uni_xy(reshape_PB[:,:,i], b)
        return loss
        
    def get_uni_xy(self, position, batch):
        pos, _ = torch.sort(position)
        M = position.shape[1]
        i = torch.arange(0., M, requires_grad=False).to(self.dev)
        return torch.mean(torch.pow(pos - i.unsqueeze(0) / (M-1),2))

    def descloss(self, DA, DB, G):
        b, c, h, w = DA.shape
        C = G <= 8
        C_ = G > 8
        # reshape_DA size = M, 256; reshape_DB size = 256, M
        AB = torch.bmm(DA.reshape((b, c,-1)).transpose(2,1), DB.reshape((b, c,-1)))
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        return torch.mean(torch.clamp(AB, min=0))

    def decorrloss(self, DA, DB):
        b, c, h, w = DA.shape
        # reshape_DA size = 256, M
        reshape_DA = DA.reshape((b,c,-1))
        # reshape_DB size = 256, M
        reshape_DB = DB.reshape((b,c,-1))
        loss = 0
        loss += self.get_R_b(reshape_DA)
        loss += self.get_R_b(reshape_DB)
        return loss
    
    def get_R_b(self, reshape_D):
        B, F = reshape_D.shape[0:2]
        v_ = torch.mean(reshape_D, dim = 2, keepdim=True)
        V_v = reshape_D - v_
        molecular = torch.matmul(V_v, V_v.transpose(2,1))
        V_v_2 = torch.sum(torch.pow(V_v, 2), dim=2, keepdim=True)
        denominator = torch.sqrt(torch.matmul(V_v_2, V_v_2.transpose(2,1)))
        one = torch.eye(F).to(self.dev).unsqueeze(0)
        rb = torch.sum(torch.square((molecular / (denominator + 1e-8) - one) / (B * F * (F-1))))
        return rb

    # New loss for changed version for URP
    def triloss(self, As, Ap, Ad, Bs, Bp, Bd, Cs, Cp, Cd,
        mat1, mat2, mask1, mask2):
        numbatch, c = Ap.shape[0:2]
        f = Ad.shape[1]

        uspA = uspB = unixy = descA = descB = decorr = 0

        position_A = self.get_position(Ap, mat=mat1)
        position_B = self.get_position(Bp, mat=mat2)
        position_C = self.get_position(Cp)

        Ga, flatmask1 = self.trigetG(position_A, position_C, mask1)
        Gb, flatmask2 = self.trigetG(position_B, position_C, mask2)

        for i in range(numbatch):
            Ga_ = Ga[i][flatmask1[i],:][:,flatmask1[i]]
            Gb_ = Gb[i][flatmask2[i],:][:,flatmask2[i]]
            uspA += self.triusploss(As[i], Cs[i], Ga_, mask1[i])
            uspB += self.triusploss(Bs[i], Cs[i], Gb_, mask2[i])

            descA += self.tridescloss(Ad[i], Cd[i], Ga_, mask1[i])
            descB += self.tridescloss(Bd[i], Cd[i], Gb_, mask2[i])
        
        uspA /= numbatch
        uspB /= numbatch
        descA /= numbatch
        descB /= numbatch

        # Uniform xy loss
        reshape_PA = Ap.reshape((numbatch,c,-1)).transpose(2,1)
        reshape_PB = Bp.reshape((numbatch,c,-1)).transpose(2,1)
        reshape_PC = Bp.reshape((numbatch,c,-1)).transpose(2,1)
        for i in range(2):
            unixy += self.get_uni_xy(reshape_PA[:,:,i], numbatch)
            unixy += self.get_uni_xy(reshape_PB[:,:,i], numbatch)
            unixy += self.get_uni_xy(reshape_PC[:,:,i], numbatch)

        # Decorrelation loss
        # reshape_DA, B, C size = 256, M
        reshape_DA = Ad.reshape((numbatch,f,-1))
        reshape_DB = Bd.reshape((numbatch,f,-1))
        reshape_DC = Cd.reshape((numbatch,f,-1))
        decorr += self.get_R_b(reshape_DA)
        decorr += self.get_R_b(reshape_DB)
        decorr += self.get_R_b(reshape_DC)

        wa = torch.sum(mask1).float()/(torch.sum(mask1) + torch.sum(mask2)).float()
        wb = 1-wa

        loss = self.usp * (wa*uspA + wb*uspB) + self.uni_xy * unixy +\
            self.desc * (wa*descA + wb*descB) + self.decorr * decorr

        lossdict = {
            "loss": loss,
            "A/usp_loss": self.usp * wa * uspA,
            "B/usp_loss": self.usp * wb * uspB,
            "uni_xy_loss": self.uni_xy * unixy,
            "A/descriptor_loss": self.desc * wa * descA,
            "B/descriptor_loss": self.desc * wb * descB,
            "decorrelation_loss": self.decorr * decorr
        }
        return lossdict
        
    def triusploss(self, As, Bs, G, mask):
        A2BId = torch.argmin(G, dim=1)
        M = A2BId.shape[0]
        Id = G.gather(1, A2BId.reshape(-1, 1)).squeeze()
        Id = Id <= self.correspond

        # print(A2BId.shape, Id.shape)
        # print(reshape_As_k.shape,reshape_Bs_k.shape,d_k.shape)
        reshape_As = As[:,mask].squeeze()
        reshape_Bs = Bs[:,mask].squeeze()
        positionK_loss = scoreK_loss = uspK_loss = 0

        d_k = G[Id, A2BId[Id]]
        reshape_As_k = reshape_As[Id]
        reshape_Bs_k = reshape_Bs[A2BId[Id]]
        positionK_loss = torch.mean(d_k)
        scoreK_loss = torch.mean(torch.pow(reshape_As_k - reshape_Bs_k, 2))
        uspK_loss = self.get_uspK_loss(d_k, reshape_As_k, reshape_Bs_k)

        return (self.position_weight * positionK_loss + 
            self.score_weight * scoreK_loss + uspK_loss)

    def trigetG(self, PA, PB, mask):
        b = PA.shape[0]
        reshape_mask = mask.reshape((b, -1))
        return self.getG(PA, PB), reshape_mask

    def tridescloss(self, Ad, Bd, G, mask):
        c = Ad.shape[0]
        C = G <= 8
        C_ = G > 8
        # Ad_reshape size = M, 256; Bd_reshape size = 256, M
        Ad_reshape = Ad[:,mask]
        Bd_reshape = Bd[:,mask]
        AB = torch.matmul(Ad_reshape.transpose(1,0), Bd_reshape)
        AB[C] = self.d * (self.m_p - AB[C])
        AB[C_] -= self.m_n
        return torch.mean(torch.clamp(AB, min=0))

    def getPtsDescFromHeatmap(self, point, heatmap, desc):
        '''
        :param self:
        :param point:
            np (2, Hc, Wc)
        :param heatmap:
            np (Hc, Wc)
        :param desc:
            np (256, Hc, Wc)
        :return:
        '''
        heatmap = heatmap.squeeze()
        desc = desc.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H = heatmap.shape[0]*self.downsample
        W = heatmap.shape[1]*self.downsample
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = point[0, xs, ys] # abuse of ys, xs
        pts[1, :] = point[1, xs, ys]
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        desc = desc[:, xs, ys]

        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        desc = desc[:, inds[::-1]]

        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        desc = desc[:, ~toremove]
        return pts[:, :300], desc[:, :300]

    def get_homography(self, Ap, Ad, Bp, Bd, As, Bs):
        Amap = self.get_position(Ap)
        Bmap = self.get_position(Bp)

        points1, points2 = self.get_match_point(Amap, Ad, Bmap, Bd, As, Bs)
        # img = cv2.imread(srcpath)
        # map = points1
        # points_list = [(int(map[i,0]),int(map[i,1])) for i in range(len(map))]
        # print(points_list)
        # for point in points_list:
        #     cv2.circle(img , point, point_size, point_color, thickness)
        # print(points1)
        # print(points2)
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return h, mask, points1, points2
    
    def get_match_point(self, Amap, Ad, Bmap, Bd, As, Bs):
        c = Amap.shape[0]
        c_d = Ad.shape[0]
        print(c,c_d)        
        reshape_As = As.reshape((-1)) 
        reshape_Bs = Bs.reshape((-1))
        reshape_Ap = Amap.reshape((c,-1)).permute(1,0)
        reshape_Bp = Bmap.reshape((c,-1)).permute(1,0)
        reshape_Ad = Ad.reshape((c_d,-1)).permute(1,0)
        reshape_Bd = Bd.reshape((c_d,-1))
        print(reshape_Ad.shape)
        D = torch.matmul(reshape_Ad,reshape_Bd)
        # print(D)
        A2B_nearest_Id = torch.argmax(D, dim=1)
        B2A_nearest_Id = torch.argmax(D, dim=0)

        print(A2B_nearest_Id)
        print(A2B_nearest_Id.shape)
        print(B2A_nearest_Id)
        print(B2A_nearest_Id.shape)

        match_B2A = B2A_nearest_Id[A2B_nearest_Id]
        A2B_Id = torch.from_numpy(np.array(range(len(A2B_nearest_Id)))).to(self.dev)

        print(match_B2A)
        print(match_B2A.shape)
        # for i in range(len(match_B2A)):
        #     print(match_B2A[i],end=' ')
        print(A2B_Id)
        print(A2B_Id.shape)

        finish_Id = A2B_Id == match_B2A
      
        points1 = reshape_Ap[finish_Id]
        points2 = reshape_Bp[A2B_nearest_Id[finish_Id]]

        return points1.cpu().numpy(), points2.cpu().numpy()

        # Id = torch.zeros_like(A2B_nearest, dtype=torch.uint8)
        # for i in range(len(A2B_nearest)):
