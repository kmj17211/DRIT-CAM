import torch
from options import TrainOptions
from dataset import dataset_unpair, Synth_Dataset, Real_Dataset
from model import DRIT
from saver import Saver

def main():

  train_sample = True

  # parse options
  parser = TrainOptions()

  if train_sample:
    parser.define_parser_defaults_SAMPLE()
  else:
    parser.define_parser_defaults()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  if train_sample:
    synth_ds = Synth_Dataset(opts)
    real_ds = Real_Dataset(opts)
    synth_dl = torch.utils.data.DataLoader(synth_ds, batch_size = opts.batch_size, shuffle = True, num_workers = opts.nThreads)
    real_dl = torch.utils.data.DataLoader(real_ds, batch_size = opts.batch_size, shuffle = True, num_workers = opts.nThreads)
  else:
    dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  ep_cycle =  max(len(synth_dl), len(real_dl))

  # train
  print('\n--- train ---')
  if train_sample:
    max_it = 500000
    num_iter = opts.n_ep * ep_cycle
    
    for it in range(num_iter):
      if not (it % len(synth_dl)):
        synth_set = iter(synth_dl)
      
      if not (it % len(real_dl)):
        real_set = iter(real_dl)

      synth_img, synth_label, _, _ = next(synth_set)
      real_img, real_label, _, _ = next(real_set)

      synth_img = synth_img.cuda(opts.gpu)
      synth_label = synth_label.cuda(opts.gpu)
      real_img = real_img.cuda(opts.gpu)
      real_label = real_label.cuda(opts.gpu)

      if (it + 1) % ep_cycle == 0:
        model.update_lr()
        saver.write_img((it+1)//ep_cycle, model)
        saver.write_model((it+1)//ep_cycle, it, model)

      if (it + 1) % opts.d_iter != 0:
        model.update_D_content(synth_img, synth_label, real_img, real_label)
        continue
      else:
        model.update_D(synth_img, synth_label, real_img, real_label)
        model.update_EG()

      if not opts.no_display_img:
        saver.write_display(it, model)
      
      print('Iter: {}/{}, lr: {}'.format(it, num_iter, model.gen_opt.param_groups[0]['lr']))
      
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, model)
        break

  else:
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
      for it, (images_a, images_b) in enumerate(train_loader):
        if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
          continue

        # input data
        images_a = images_a.cuda(opts.gpu).detach()
        images_b = images_b.cuda(opts.gpu).detach()

        # update model
        if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
          model.update_D_content(images_a, images_b)
          continue
        else:
          model.update_D(images_a, images_b)
          model.update_EG()

        # save to display file
        if not opts.no_display_img:
          saver.write_display(total_it, model)

        print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
        total_it += 1
        if total_it >= max_it:
          saver.write_img(-1, model)
          saver.write_model(-1, model)
          break

      # decay learning rate
      if opts.n_ep_decay > -1:
        model.update_lr()

      # save result image
      saver.write_img(ep, model)

      # Save network weights
      saver.write_model(ep, total_it, model)

  return

if __name__ == '__main__':
  main()
