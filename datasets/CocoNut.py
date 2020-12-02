import cv2
import os
import random
import math

from PIL import Image
import numpy as np
from pathlib import Path

import torch
from datasets.Coco import Coco
from utils.tools import dict_update, Myresize
from utils.utils import sample_homography_np as sample_homography
from utils.utils import inv_warp_image, warp_image_np

from settings import COCO_TRAIN, COCO_VAL, DATA_PATH

class CocoNut(Coco):
    '''
        New dataset class based on KAIST URP
    '''
    def __init__(self, transform=None, task='train', **config):
        super(CocoNut, self).__init__(transform, task, **config)

        root = Path(DATA_PATH, 'COCO/' + task + '2014/')
        images = list(root.iterdir())
        self.images = [[str(p), str(q)] for p, q in zip(images, random.sample(images, len(images)))]

    def __getitem__(self, index):
        img1_path, img2_path = self.images[index]
        cv_img1 = cv2.imread(img1_path)
        cv_img2 = cv2.imread(img2_path)
        re_img1 = Myresize(cv_img1, self.config['resize'])
        re_img2 = Myresize(cv_img2, self.config['resize'])
        tran_img1 = self.EnhanceData(re_img1)
        tran_img2 = self.EnhanceData(re_img2)

        if self.transforms:
            re_img1 = Image.fromarray(cv2.cvtColor(re_img1, cv2.COLOR_BGR2RGB))
            source_img1 = self.transforms(re_img1)

            re_img2 = Image.fromarray(cv2.cvtColor(re_img2, cv2.COLOR_BGR2RGB))
            source_img2 = self.transforms(re_img2)

            tran_img1 = Image.fromarray(cv2.cvtColor(tran_img1, cv2.COLOR_BGR2RGB))
            tran_img1 = self.transforms(tran_img1)

            tran_img2 = Image.fromarray(cv2.cvtColor(tran_img2, cv2.COLOR_BGR2RGB))
            tran_img2 = self.transforms(tran_img2)

        mat1 = sample_homography(np.array([2, 2]), shift=-1, **self.config['homographies'])
        mat2 = sample_homography(np.array([2, 2]), shift=-1, **self.config['homographies'])

        # Mask resampler
        unw_bmask = np.ones(self.config['resize'], dtype=float)
        # mask = self.generate_mask(self.config['resize'])
        mask = self.square_mask(self.config['resize'])
        mask = warp_image_np(mask, mat2)
        cmask = unw_bmask - mask
        bmask = warp_image_np(unw_bmask, mat1)

        while np.sum(bmask*mask)/np.sum(bmask*(cmask + mask)) < 0.2:
            mask = self.square_mask(self.config['resize'])
            mask = warp_image_np(mask, mat2)
            cmask = unw_bmask - mask
            bmask = warp_image_np(unw_bmask, mat1)

        mask = torch.tensor(mask, dtype=torch.float32)
        cmask = torch.tensor(cmask, dtype=torch.float32)
        bmask = torch.tensor(bmask, dtype=torch.float32)

        mat1 = torch.tensor(mat1, dtype=torch.float32)
        mat2 = torch.tensor(mat2, dtype=torch.float32)

        inv_mat1 = torch.inverse(mat1)
        tran_img1 = inv_warp_image(tran_img1, inv_mat1).squeeze(0)

        inv_mat2 = torch.inverse(mat2)
        tran_img2 = inv_warp_image(tran_img2, inv_mat2).squeeze(0)

        des_img = bmask*(cmask*tran_img1 + mask*tran_img2)

        if torch.isnan(source_img1).any():
            print("NAN is corrected in src image")
            des_img[torch.isnan(source_img1)] = 0.

        if torch.isnan(source_img2).any():
            print("NAN is corrected in src image")
            des_img[torch.isnan(source_img2)] = 0.

        if torch.isnan(des_img).any():
            print("NAN is corrected in des image")
            des_img[torch.isnan(des_img)] = 0.

        return source_img1, source_img2, des_img, mat1, mat2, bmask*cmask, bmask*mask
        # return source_img1, des_img, mat1

    def generate_mask(self, size=[240, 320]):
        '''
            Given size, make a mask with random shape patches
        '''
        # Prepare a canvas
        mask = np.zeros([size[0], size[1], 3], np.uint8)

        numpatches = random.randint(3, 8)
        lenmean = min(size)
        for i in range(numpatches):
            direction = -math.pi/2
            pts = [(random.randrange(size[1]), random.randrange(size[0]))]
            m = random.randint(2, 6)
            for j in range(m):
                length = random.uniform(lenmean*0.075, lenmean*0.125)
                direction += random.uniform(0, math.pi/2)
                vector = (int(length * math.cos(direction)) + pts[j][0],
                          int(length * math.sin(direction)) + pts[j][1])

                pts.append(vector)

            # Draw a polygon
            cv2.fillPoly(mask, [np.array(pts, np.int32)], (255, 255, 255))

        # return torch.tensor(mask[:, :, 0]/255, dtype=torch.float32)
        return(mask[:, :, 0]/255).astype(np.float32)

    def square_mask(self, size=[240, 320]):
        '''
            Given size, make a mask with square patches
        '''
        # Prepare a canvas
        mask = np.zeros([size[0], size[1], 3], np.uint8)
        while np.sum(mask)/(size[0]*size[1]) < 0.3:
            numsq = random.randint(3, 6)
            lenmean = min(size)
            for i in range(numsq):
                pts = [(random.randrange(size[1]), random.randrange(size[0]))]
                length = random.uniform(lenmean*0.175, lenmean*0.225)
                length = lenmean*0.5
                direction = random.uniform(0, 2*math.pi)
                pts.append((int(length * math.cos(direction)) + pts[0][0],
                            int(length * math.sin(direction)) + pts[0][1]))
                vector = [length * math.cos(direction), length * math.sin(direction)]

                length = random.uniform(lenmean*0.175, lenmean*0.225)
                length = lenmean*0.5
                direction += random.uniform(math.pi/6, math.pi/2)
                pts.append((int(length * math.cos(direction)) + pts[0][0],
                    int(length * math.sin(direction)) + pts[0][1]))
                
                pts.append((int(vector[0]) + pts[2][0], int(vector[1]) + pts[2][1]))
                tmp = pts[2]
                pts[2] = pts[3]
                pts[3] = tmp
                
                # Draw a polygon
                cv2.fillPoly(mask, [np.array(pts, np.int32)], (255, 255, 255))

        # return torch.tensor(mask[:, :, 0]/255, dtype=torch.float32)
        return (mask[:, :, 0]/255).astype(np.float32)