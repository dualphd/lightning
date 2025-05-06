import os, glob
import time
import argparse
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from mymodel import mygen_model
from myutils import init_weight, ImagePool, LossDisplayer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

from mydataset import give_me_dataloader, PairedDataset, give_me_transform
from myloss import BayerLoss
import matplotlib.pyplot as plt

def save_model(model, ckpt_path, epoch, loss=0.0, state='valid'):
    try:
        os.makedirs(ckpt_path, exist_ok=True)
        fname = os.path.join(ckpt_path, "bwunet_epoch_%05d_loss_%05.3e.pth"%(epoch, loss))
        if os.path.exists(fname):
            fname = fname.split('.pth')[0] + f'_{state}_1.pth'
        print('trying to save,,,,', fname)
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch"      : epoch,
                },
                fname,
        )
    except:
        print('something wrong......skip saving model at epoch ', epoch)


def give_me_visualization(model, dataloader, device):
    pbar = tqdm(dataloader)
    for idx, pairs in enumerate(pbar):
        ## for tensorboard viz

        model.eval()

        # get data
        item_in, item_gt = pairs
        if device== 'mps':
            item_in = item_in.type(torch.float32)
            item_gt = item_gt.type(torch.float32)
        item_in = item_in.to(device)
        item_gt = item_gt.to(device)

        # forward
        item_out = model(item_in)

        # diff
        item_diff = torch.abs(item_out)

        ## normalize for grid view
        item_in = (item_in + 1) / 2
        item_gt = (item_gt + 1) / 2
        item_diff = (item_diff + 2) / 4
        item_out = (item_out + 1) / 2

        # get grid view
        item_in = vutils.make_grid(item_in, padding=2, normalize=True)
        item_gt = vutils.make_grid(item_gt, padding=2, normalize=True)
        item_diff = vutils.make_grid(item_diff, padding=2, normalize=True)
        item_out = vutils.make_grid(item_out, padding=2, normalize=True)

        top_images = torch.cat((item_in.cpu(), item_gt.cpu()), dim=2)
        bot_images = torch.cat((item_diff.cpu(), item_out.cpu()), dim=2)
        viz_images = torch.cat((top_images.cpu(), bot_images.cpu()), dim=1)

        return viz_images


def train(args):

    bwtest = True

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Train Using {device} device")
    print(args)

    # args
    model_name      = 'bwunet'
    dataset_path    = '/Users/bw/Dataset/MIPI_demosaic_hybridevs'
    input_size      = 256
    batch_size      = 128
    learning_rate   = 1e-4
    checkpoint_path = 'model_dir_torch/ckpt'

    print('model_name = ', model_name)
    print('input_size = ', input_size)
    print('device = ', device)
    print('dataset_path = ', dataset_path)

    # dataset - train, valid, viz
    base_path = os.path.join(dataset_path)
    print('base_path: ', base_path)
    pnames_in = glob.glob(os.path.join(base_path, 'train/pairs256', "*_in.npy"))
    pnames_gt = glob.glob(os.path.join(base_path, 'train/pairs256', "*_gt.npy"))
    pnames_viz_in = glob.glob(os.path.join(base_path, 'viz/pairs256', "*_in.npy"))
    pnames_viz_gt = glob.glob(os.path.join(base_path, 'viz/pairs256', "*_gt.npy"))


    pnames_in.sort()
    pnames_gt.sort()
    pnames_viz_in.sort()
    pnames_viz_gt.sort()


    flen = len(pnames_gt)
    print(flen)

    order = np.arange(flen)
    np.random.shuffle(order)
    print(order)

    pnames_in = [pnames_in[x] for x in order]
    pnames_gt = [pnames_gt[x] for x in order]

    validation_split = 0.05
    number_train_set = int(flen*(1-validation_split))
    number_valid_set = flen - number_train_set
    print("-->>>>",number_train_set, number_valid_set)

    pnames_train_in = pnames_in[:number_train_set]
    pnames_train_gt = pnames_gt[:number_train_set]

    pnames_valid_in = pnames_in[number_train_set:]
    pnames_valid_gt = pnames_gt[number_train_set:]

    if bwtest: # for test purpose
        pnames_train_in = pnames_in[:16]
        pnames_train_gt = pnames_gt[:16]

        pnames_valid_in = pnames_in[16:32]
        pnames_valid_gt = pnames_gt[16:32]


    # transform
    transform = {'train': give_me_transform('train'),
                 'valid': give_me_transform('valid'),
                 'viz':   give_me_transform('viz')}

    # dataloader
    dataloader = {'train': give_me_dataloader(PairedDataset(pnames_train_in, pnames_train_gt, transform['train'], device), batch_size),
                  'valid': give_me_dataloader(PairedDataset(pnames_valid_in, pnames_valid_gt, transform['valid'], device), batch_size),
                  'viz'  : give_me_dataloader(PairedDataset(pnames_viz_in,   pnames_viz_gt,   transform['viz'],   device), batch_size) }

    nsteps={}
    for state in ['train', 'valid', 'viz']:
        nsteps[state] = len(dataloader[state])
        print('len(%s): '%state, len(dataloader[state]))

    # model
    model = mygen_model('bwunet').to(device)
    print(model)


    # save onnx
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device, requires_grad=False)
    with torch.no_grad():
        os.makedirs('onnx', exist_ok=True)
        torch.onnx.export(model.eval(), dummy_input,
                          os.path.join('onnx',  f"{model_name}.onnx"))


    # ckpt save load if any
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_list = os.listdir(checkpoint_path)
    ckpt_list.sort()

    if (checkpoint_path is not None) and \
            os.path.exists(checkpoint_path) and \
            (len(ckpt_list)) > 0:
        ckpt_name = os.path.join(checkpoint_path, ckpt_list[-1])
        print('Loading.....', ckpt_name)
        checkpoint = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
    else:
        os.makedirs(checkpoint_path, exist_ok=True)
        epoch = 0

    # make train mode
    model.train()

    # # Loss
    # criterion_bayer = BayerLoss()
    # criterion_cycle = nn.L1Loss()
    # criterion_identity = nn.L1Loss()
    criterion = nn.MSELoss()



    # Optimizer, Schedular
    optim_G = optim.Adam(
        list(model.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )


    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)


    # Training
    # logger for tensorboard.
    logpath = os.path.join('model_dir_torch', 'board')
    os.makedirs(logpath, exist_ok=True)
    summary = SummaryWriter(logpath)
    disp_train = LossDisplayer(["G_train"])
    disp_valid = LossDisplayer(["G_valid"])
    disp = {'train':disp_train, 'valid':disp_valid}

    step = {'train':epoch*nsteps['train'], 'valid':epoch*nsteps['train'], 'viz':epoch*nsteps['train']}

    loss_best_G = {'train': float('inf'), 'valid': float('inf')}
    loss_G_train_last = float('inf')


    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch}")

        loss_G_total = {'train': 0, 'valid': 0}
        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, pairs in enumerate(pbar):
                pbar.set_description('Processing %s...  epoch %d' % (state, epoch))

                if state == 'train' and idx == 0:
                    # train mode
                    model.train()
                elif state == 'valid' and idx == 0:
                    # eval mode
                    model.eval()


                # get data
                item_in, item_gt = pairs

                if device == 'mps':
                    item_in = item_in.type(torch.float32)
                    item_gt = item_gt.type(torch.float32)

                # data to device
                item_in = item_in.to(device)
                item_gt = item_gt.to(device)



                # -----------------
                # Forward
                # -----------------
                item_out = model(item_in)

                # -----------------
                # Train Generator
                # -----------------
                loss_mse = criterion(item_out, item_gt)

                # combine loss and calculate gradients
                loss_G = 0
                loss_G += loss_mse
                loss_G_train_last = loss_G  # for save

                step[state] += 1
                if state == 'train':
                    # train mode
                    optim_G.zero_grad()
                    loss_G.backward()
                    optim_G.step()
                else:
                    if idx == 0:
                        step['valid'] = step['train']

                ## accumulate generator loss in validataion to save best ckpt
                loss_G_total[state] += loss_G

                # -----------------
                # record loss for tensorboard
                # -----------------
                disp[state].record([loss_G])
                # if step[state] % 1 == 0 and idx>1:
                if True:
                    avg_losses = disp[state].get_avg_losses()
                    summary.add_scalar(f"loss_G_{state}", avg_losses[0], step[state])

                    print(
                        f'{state} : epoch{epoch}, step{step[state]}------------------------------------------------------')
                    print('loss_G: %.3f, ' % avg_losses[0], end='')
                    disp[state].reset()


        else:
            print('hello<<< ', state)

            viz_images = give_me_visualization(model, dataloader['viz'], device )

            summary.add_image('generated pairs', viz_images, step['viz'] )


            ## save ckeck point if improved
            loss_G_average = loss_G_total[state] / nsteps[state]
            if loss_best_G[state] > loss_G_average:
                print(f'best {state} ckpt updated!!!  old best {loss_best_G[state]} vs new best {loss_G_average}')
                loss_best_G[state] = loss_G_average
                summary.add_scalar(f"loss_best_G{state}", loss_best_G[state], step[state])

                ckpt_path_name_best = os.path.join(checkpoint_path, model_name + '_best')

                save_model(model, ckpt_path_name_best, epoch, loss_best_G[state])

        ## Step scheduler
        scheduler_G.step()

        # Save checkpoint for every 5 epoch
        if epoch % 5 == 0:
            save_model(model, checkpoint_path, epoch, loss_G_train_last)

    print('done done')






def main(args):
    train(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--model_name', default='resnet', type=str,
                    choices=['resnet', 'unet', 'bwunet'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_name', default='div2k', type=str,
                    choices=['div2k'],
                    help='(default=%(default)s)')
    argparser.add_argument('--dataset_path',
                           default='/Users/bw/Dataset/MIPI_demosaic_hybridevs/train/pairs', type=str,
                    help='(default=datasets)')
    argparser.add_argument('--checkpoint_path', default=f"model_dir_torch/ckpt",
                    type=str, help='(default=%(default)s)')
    argparser.add_argument('--device', default='mps', type=str,
                    choices=['cpu','cuda', 'mps'],
                    help='(default=%(default)s)')
    argparser.add_argument('--input_size', type=int, help='input size', default=128)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='mini batch size', default=2)
    args = argparser.parse_args()
    main(args)