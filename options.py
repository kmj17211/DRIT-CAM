import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt
  
  def define_parser_defaults(self):
    self.parser.set_defaults(dataroot = './Data/yosemite',
                             phase = 'train',
                             batch_size = 2,
                             resize_size = 256,
                             crop_size = 216,
                             input_dim_a = 3,
                             input_dim_b = 3,
                             nThreads = 8,
                             name = 'yosemite',
                             display_dir = './Weight/DRIT++/yosemite/logs',
                             result_dir = './Weight/DRIT++/yosemite/results',
                             display_freq = 1,
                             img_save_freq = 5,
                             model_save_freq = 10,
                             concat = 1,
                             dis_scale = 3,
                             dis_norm = 'None',
                             lr_policy = 'lambda',
                             n_ep = 1200,
                             n_ep_decay = 600,
                             resume = None,
                             d_iter = 3,
                             gpu = 0)
    
  def define_parser_defaults_SAMPLE(self):
    self.parser.set_defaults(dataroot = './Data/SAR Data/SAMPLE/mat_files_pix2pix_60%',
                             phase = 'train',
                             batch_size = 2,
                             resize_size = 128,
                             crop_size = 128,
                             input_dim_a = 1,
                             input_dim_b = 1,
                             nThreads = 8,
                             name = 'SAMPLE',
                             display_dir = './Weight/DRIT++CAM/SAMPLE_10_10class/logs',
                             result_dir = './Weight/DRIT++CAM/SAMPLE_10_10class/results',
                             display_freq = 100,
                             img_save_freq = 5,
                             model_save_freq = 10,
                             concat = 1,
                             dis_scale = 3,
                             dis_norm = 'None',
                             lr_policy = 'lambda',
                             n_ep = 1200,
                             n_ep_decay = 600,
                             resume = None,
                             d_iter = 3,
                             gpu = 0)

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--resume', type=str, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt

  def define_parser_defaults(self):
    self.parser.set_defaults(dataroot = './Data/yosemite',
                             phase = 'test',
                             resize_size = 256,
                             crop_size = 216,
                             input_dim_a = 3,
                             input_dim_b = 3,
                             nThreads = 8,
                             a2b = 1,
                             name = 'yosemite',
                             num = 5,
                             result_dir = './Weight/DRIT++/yosemite/results',
                             concat = 1,
                             dis_scale = 3,
                             resume = './Weight/DRIT++/yosemite/results/yosemite/01199.pth',
                             gpu = 0)
    
  def define_parser_defaults_SAMPLE(self):
    self.parser.set_defaults(dataroot = './Data/SAR Data/SAMPLE/mat_files_pix2pix_60%',
                             phase = 'test',
                             resize_size = 128,
                             crop_size = 128,
                             input_dim_a = 1,
                             input_dim_b = 1,
                             nThreads = 8,
                             a2b = 1,
                             name = 'SAMPLE',
                             num = 5,
                             result_dir = './Weight/DRIT++CAM/SAMPLE_10_10class/results',
                             concat = 1,
                             dis_scale = 3,
                             resume = './Weight/DRIT++CAM/SAMPLE_10_10class/results/SAMPLE/00299.pth',
                             gpu = 0)