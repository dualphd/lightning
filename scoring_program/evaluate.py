#!/usr/bin/env python
import numpy as np
import os.path
import sys
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io, color


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension=False):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(x_true[:, :, i], x_pred[:, :, i], data_range=data_range)
             for i in range(x_true.shape[2])]

    return np.mean(mssim)


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref/gt')

output_filename = os.path.join(output_dir, 'scores.txt')

print(submit_dir)
print(truth_dir)
print(output_dir)

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Get the path of gt file
img_list = []
# file_list = sorted(os.listdir(submit_dir))
file_list = sorted(os.listdir(truth_dir))

# set up the metris you need.
PSNRs = []
SSIMs = []

for item in file_list:
    res_img = io.imread(os.path.join(submit_dir, item))
    ref_img = io.imread(os.path.join(truth_dir, item))

    assert res_img.dtype == np.uint8 and ref_img.dtype == np.uint8, 'RGB images should be of type uint8'
    res_img = res_img.astype(np.float32)
    ref_img = ref_img.astype(np.float32)
    PSNR = compare_mpsnr(res_img, ref_img, data_range=255)
    SSIM = compare_mssim(res_img, ref_img, data_range=255)
    PSNRs.append(PSNR)
    SSIMs.append(SSIM)

score_psnr = np.mean(PSNRs)
score_ssim = np.mean(SSIMs)

# Write the result into score_path/score.txt
with open(output_filename, 'w') as f:
    f.write('{}: {}\n'.format('PSNR', score_psnr))
    f.write('{}: {}\n'.format('SSIM', score_ssim))
    f.write('DEVICE: CPU\n')
