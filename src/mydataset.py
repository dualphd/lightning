
import os
import numpy as np
import torch
import glob, random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


crop_size = 256
cfa_pattern = 2
idx_R = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G1 = np.tile(
        np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G2 = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_B = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_G = np.tile(
        np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))


idx_RGB = np.concatenate((idx_R[np.newaxis, ...],
                          idx_G[np.newaxis, ...],
                          idx_B[np.newaxis, ...]), axis=0)

idx_G1RBG2 = np.concatenate((idx_G1[np.newaxis, ...],
                             idx_R [np.newaxis, ...],
                             idx_B [np.newaxis, ...],
                             idx_G2[np.newaxis, ...]), axis=0)

def get_nonshrink_from_1ch(image):
    image_R = image * idx_R
    image_G = image * idx_G
    image_B = image * idx_B

    image = np.stack((image_R, image_G, image_B), axis=-1)

    return image





def give_me_test_images(input_size=256):
    fname = ['apple.jpg', 'orange.jpg']
    image_tensor = torch.zeros(len(fname), 3, input_size, input_size)
    # image_list = []
    for idx, f in enumerate(fname):
        fpath = os.path.join('imgs', f)
        image = torch.Tensor(np.array(Image.open(fpath))).detach()
        image_tensor[idx] = image.permute(2,0,1)
    return image_tensor


def give_me_comparison(model, inputs, device):
    # print('inputs.size ', inputs.size(), type(inputs))
    with torch.no_grad():
        model.eval()
        if device == torch.device('cuda'):
            inputs=inputs.cuda()
            # print('input is in cuda')
        else:
            # print('input is in cpu')
            ...

        # print(type(inputs))
        # model.cpu()
        if device=='cuda' or next(model.parameters()).is_cuda:
            inputs=inputs.cuda()
        outputs = model(inputs)
    return outputs



def give_me_visualization(model_A2B, model_B2A=None, device='cpu', test_batch=None, nomalize=True, beta_for_gamma=2.2):
    # visualize test images
    # print('test_batch', type(test_batch))
    if test_batch != None:
        real_B_images = test_batch.cpu() # RAW
    else:
        real_B_images = give_me_test_images().to(device)
    real_A_images = gamma(real_B_images, device, beta_for_gamma) # RGB
    fake_B_images = give_me_comparison(model_A2B, real_A_images.to(device), device=device)
    if model_B2A == None:
        # fake_rgb_images = torch.zeros_like(real_raw_images)
        fake_A_images = torch.abs(real_B_images.to(device) - fake_B_images.to(device)) ## diff when pix2pix

    else:
        fake_A_images = give_me_comparison(model_B2A, real_B_images.to(device), device=device)

    print('real_A (%.3f, %.3f), ' %(torch.amin(real_A_images), torch.amax(real_A_images)), end='')
    print('real_B (%.3f, %.3f), ' %(torch.amin(real_B_images), torch.amax(real_B_images)), end='')
    print('fake_A (%.3f, %.3f), ' %(torch.amin(fake_A_images), torch.amax(fake_A_images)), end='')
    print('fake_B (%.3f, %.3f), ' %(torch.amin(fake_B_images), torch.amax(fake_B_images)))

    real_A_images = vutils.make_grid(real_A_images, padding=2, normalize=nomalize)
    real_B_images = vutils.make_grid(real_B_images, padding=2, normalize=nomalize)
    fake_A_images = vutils.make_grid(fake_A_images, padding=2, normalize=nomalize)
    fake_B_images = vutils.make_grid(fake_B_images, padding=2, normalize=nomalize)

    real_images = torch.cat((real_A_images.cpu(), real_B_images.cpu() ), dim=2)
    fake_images = torch.cat((fake_A_images.cpu(), fake_B_images.cpu() ), dim=2)
    test_images = torch.cat((real_images.cpu(),   fake_images.cpu()),    dim=1)

    # if test_batch != None:
    test_images = test_images.permute(1,2,0)
    return test_images




# trandform
def give_me_transform(type, mean=0.5, std=0.5):

    transform = None
    if type == 'train':
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((args.size, args.size)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    return transform


class Flip:
    def __init__(self, axis):
        self.axis = axis
    def __call__(self, x):
        # print('x.shape', x.shape)
        return np.flip(x, self.axis).copy()

class Rot90:
    def __init__(self):
        ...
    def __call__(self, x):
        return np.rot90(x, np.random.randint(4)).copy()

class Normalization():
    def __init__(self, bits=16, mean=0.5, std=0.5):
        self.maxval = (2**bits) -1
        self.mean = mean
        self.std = std
    def __call__(self, x):
        x = x.astype(np.float32) / self.maxval
        x = (x - self.mean) / self.std
        return x.copy()


# dataloader
def give_me_dataloader(dataset, batch_size:int, shuffle=True, num_workers=2, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


# dataset
class SingleDataset(DataLoader):
    def __init__(self, dataset_dir, transforms, mylen=-1, bits=8):
        self.dataset_dir = dataset_dir
        self.transform = transforms
        self.mylen = mylen
        self.bits = bits

        self.image_path = glob.glob(os.path.join(dataset_dir, "**/*") , recursive=True)
        if mylen>0:
            self.image_path = self.image_path[:mylen]

        print('--------> # of images: ', len(self.image_path))

    def __getitem__(self, index):
        if self.image_path[index].split('.')[-1] == 'npy':
            item = np.load(self.image_path[index])
            item = self.transform(item).transpose(2, 0, 1)
        else:
            item = self.transform(Image.open(self.image_path[index]))
        return item

    def __len__(self):
        return len(self.image_path)




class PairedDataset(DataLoader):
    def __init__(self, pnames_in, pnames_gt, transforms, device):

        self.pnames_in = pnames_in
        self.pnames_gt = pnames_gt

        self.transform = transforms

        self.device = device

    def __getitem__(self, index):
        item_in = np.load(self.pnames_in[index])
        item_gt = np.load(self.pnames_gt[index])

        # item_in = self.transform(item_in).transpose(2, 0, 1)

        # CxHxW
        item_in = self.get_nonshrink_from_1ch(item_in) #.transpose(2,0,1)
        item_gt = item_gt #.transpose(2,0,1)

        # normalizse
        item_in = ((item_in / 1023.) * 2) - 1
        item_gt = ((item_gt /  255.) * 2) - 1

        # cure static bp in in
        item_in = self.cure_static_bp(item_in)

        # ## transforms
        seed = np.random.randint(8)
        item_in = self.mytransform(item_in, seed)
        item_gt = self.mytransform(item_gt, seed)

        ## permute
        item_in = item_in.transpose(2, 0, 1)
        item_gt = item_gt.transpose(2, 0, 1)

        return [item_in, item_gt]
    def cure_static_bp(self, item):

        # inp = tf.make_ndarray(inp)
        ## Red
        for yy in range(1,item.shape[0],4):
            for xx in range(1, item.shape[1], 4):
                item[yy][xx] = ((item[yy-1][xx]+item[yy][xx-1])/2)

        ## Blue
        for yy in range(3,item.shape[0],4):
            for xx in range(3, item.shape[1], 4):
                item[yy][xx] = ((item[yy-1][xx]+item[yy][xx-1])/2)

        return item
    def mytransform(self, image, seed):
        rot = seed & 3
        isMirror = (seed>>2) & 1
        isFlip = (seed>>3)

        image = np.rot90(image, rot).copy()

        if isMirror:
            image = np.fliplr(image).copy()
        if isFlip:
            image = np.flipud(image).copy()

        return image

    def __len__(self):
        return len(self.pnames_in)

    def get_nonshrink_from_1ch(self, image):
        image_R = image * idx_R
        image_G = image * idx_G
        image_B = image * idx_B

        image = np.stack((image_R, image_G, image_B), axis=-1)

        return image











def main():
    # degamma_example()
    # gamma_example()
    pass



if __name__ == '__main__':
    main()
