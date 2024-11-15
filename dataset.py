import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import torch
from os.path import join
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class Real_Dataset(Dataset):
    def __init__(self, args, train = True, ep = 1e-3):
        super(Real_Dataset).__init__()

        path = args.dataroot + '/real'

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5),
                                            transforms.RandomVerticalFlip()])

        self.train = train
        self.ep = ep
            
        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if train:
            self.dir_t = 'train'
            # self.label = self.label[:7]
            self.label_blk = self.label #+ ['blk1', 'blk2', 'blk3']
        else:
            self.dir_t = 'test'

        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)

    def __getitem__(self, index):
        
        # Label
        label_str = self.data_label[index]
        label = torch.zeros(len(self.label_blk))
        label[self.label_blk.index(label_str)] = 1

        # File Name
        file_name = self.file_name[index]

        # Logarithm
        img = abs(loadmat(self.data_path[index])['complex_img'])
        img = np.log10(img + self.ep)
        img = (img - np.log10(self.ep)) / (np.max(img) - np.log10(self.ep))

        img = self.transform(img)

        return img.type(torch.float32), label, file_name, label_str
    
    def __len__(self):
        return len(self.data_path)
    

class Synth_Dataset(Dataset):
    def __init__(self, args, train = True, ep = 1e-3):
        super(Synth_Dataset).__init__()

        path = args.dataroot + '/synth'

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
        
        self.train = train
        self.ep = ep

        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if train:
            self.dir_t = 'train'
            # self.label = self.label[3:]
            self.label_blk = ['blk1', 'blk2', 'blk3'] + self.label
            self.label_blk = self.label
        else:
            self.dir_t = 'test'
            self.label_blk = self.label


        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)

    def __getitem__(self, index):
        
        # Label
        label_str = self.data_label[index]
        label = torch.zeros(len(self.label_blk))
        label[self.label_blk.index(label_str)] = 1

        # File Name
        file_name = self.file_name[index]
        file_name = file_name.replace('synth', 'refine')

        # Logarithm
        img = abs(loadmat(self.data_path[index])['complex_img'])
        img = np.log10(img + self.ep)
        img = (img - np.log10(self.ep)) / (np.max(img) - np.log10(self.ep))

        img = self.transform(img)

        return img.type(torch.float32), label, file_name, label_str
    
    def __len__(self):
        return len(self.data_path)
    
