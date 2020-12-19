"""
    New test dataset for KAIST URP
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset
from utils.tools import dict_update

from settings import DATA_PATH

class SceneCity(Dataset):
    default_config = {
        'dataset': 'scenecity',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, transform=None, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.files = self._init_dataset(**self.config)

        sequence_set = []
        for (img, img_warped) in zip(self.files['image_paths'], self.files['warped_image_paths']):
            sample = {'image': img, 'warped_image': img_warped, 'fundamental': img.split('/')[2]}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform

        for k in self.files['fundamentals'].keys():
            s = config['preprocessing']['factor']
            K = self.files['fundamentals'][k][:3,:]
            E = self.files['fundamentals'][k][3:,:]
            K[2, 2] = 1/s
            K *= s
            K_inv = np.linalg.inv(K)
            self.files['fundamentals'][k] = np.dot(K_inv.T, np.dot(E, K_inv))

        if config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])

        self.fundamentals = self.files['fundamentals']

    def __getitem__(self, index):
        """
        :param index:
        :return:
            image:
                tensor (1,H,W)
            warped_image:
                tensor (1,H,W)
        """
        def _read_image(path):
            input_image = cv2.imread(path)
            return input_image

        def _preprocess(image):
            image = cv2.resize(image, (self.sizer[1], self.sizer[0]))
            if image.ndim == 2:
                image = image[:,:, np.newaxis]
            
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                image = self.transform(image)
            return image

        sample = self.samples[index]
        image = _preprocess(_read_image(sample['image']))
        warped_image = _preprocess(_read_image(sample['warped_image']))
        fundamental = self.fundamentals[sample['fundamental']]

        to_numpy = False
        if to_numpy:
            image, warped_image = np.array(image), np.array(warped_image)

        sample = {'image': image, 'warped_image': warped_image, 'fundamental': fundamental}
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, config['dataset'])
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        fundamentals = {}
        for path in folder_paths:
            file_ext = '.png'
            lpath = Path(path, 'left')
            lpath = [x for x in lpath.iterdir() if str(x).endswith(file_ext)]
            rpath = [str(x).split('/')[-1] for x in lpath]
            rpath = [Path(path, 'right', x) for x in rpath]
            for limg, rimg in zip(lpath, rpath):
                image_paths.append(str(limg))
                warped_image_paths.append(str(rimg))

            # Read Fundamental Matrix (Original scale)
            fundamentals.update({str(path).split('/')[-1]:np.loadtxt(str(Path(path, 'fundamental.txt')), delimiter=',')})

        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'fundamentals': fundamentals}
        return files


