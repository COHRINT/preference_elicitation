import pickle as pk
import sys
import random

import torchvision.transforms.functional
from torch.utils.data import Dataset
from matplotlib import image
from PIL import Image as PIL_image
import numpy as np
import os
from torchvision.io import read_image
from torchvision import transforms
import glob


############################################################
### IO
############################################################
def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()


def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious


def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()

############################################################
### IO
############################################################


def get_image_metrics(filename):
    im = PIL_image.open(filename)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    im_tr = transform(im)
    mean, std = im_tr.mean([1, 2]), im_tr.std([1, 2])
    return mean, std


def sample_image(filename, save_path, set_name, n_samples, sample_size):
    """Randomly samples and saves images to a given directory."""
    data = image.imread(filename)
    x_avail = data.shape[0] - sample_size - 1
    y_avail = data.shape[1] - sample_size - 1
    x_set = range(0, x_avail)
    y_set = range(0, y_avail)

    for s in range(0, n_samples, 3):
        x_sample = random.choice(x_set)
        y_sample = random.choice(y_set)
        new_img = data[x_sample:x_sample + sample_size, y_sample:y_sample + sample_size]
        path = str(save_path + set_name + str(s) + ".jpeg")
        # print(path)
        image.imsave(path, new_img)
        path_flipped = str(save_path+set_name+str(s+1)+".jpeg")
        image.imsave(path_flipped, np.flipud(new_img))
        path_mirror = str(save_path+set_name+str(s+2)+".jpeg")
        image.imsave(path_mirror, np.fliplr(new_img))


def normalize_image(img):
    # MapBuilder Topo
    # mean = [0.6872, 0.6945, 0.6521]
    # std = [0.1499, 0.1589, 0.1614]
    # TF Outdoors
    # mean = [0.7487, 0.8285, 0.7654]
    # std = [0.0841, 0.0788, 0.0951]
    # Forest Service Raw
    mean = [0.9133, 0.9372, 0.8405]
    std = [0.0963, 0.1203, 0.1381]
    im = torchvision.transforms.functional.normalize(img, mean, std)
    return im

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, name, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.data_name = name
        self.transform = transform
        # self.transform = transforms.ToTensor()
        self.target_transform = target_transform

    def __len__(self):
        l_files = len(glob.glob(self.img_dir+"/*"))
        # Logic helps Windows/Linux path compatibility
        if l_files == 0:
            self.img_dir = self.img_dir[1:]
            return len(glob.glob(self.img_dir+"/*"))
        else:
            return l_files

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.data_name+str(idx)+'.jpeg'))
        im = PIL_image.open(img_path)
        im = torchvision.transforms.functional.to_tensor(im)
        im = normalize_image(im)
        # im = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        # im = read_image(img_path)
        # im = im.float()
        # im = torchvision.transforms.functional.to_tensor(im)
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            label = self.target_transform()
        return im

