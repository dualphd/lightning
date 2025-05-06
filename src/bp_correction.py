import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import random




crop_size = 128
cfa_pattern = 2
idx_R = np.tile(
        np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))



idx_G = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_B = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                        np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

idx_RGB = np.concatenate((idx_R[ ..., np.newaxis],
                          idx_G[..., np.newaxis],
                          idx_B[..., np.newaxis]), axis=-1)



def cure_dynamic_bp(arr_patch):
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O

    print(arr_patch.shape, idx_R.shape)

    crop_size = 128
    cfa_pattern = 2
    idx_C = np.tile(
        np.concatenate(
            (np.concatenate((np.ones((2, 2)), np.zeros((2, 2))), axis=1),
                   np.concatenate((np.zeros((2, 2)), np.zeros((2, 2))), axis=1)),  axis=0),
        ( 2 ,  2 ))

    # return arr_patch

    # print('-->', idx_R>0)
    for yy in range(0, arr_patch.shape[0]-8, 2):
        for xx in range(0, arr_patch.shape[1]-8, 2):
            sy = yy
            ey = sy+8
            sx = xx
            ex = sx+8
            patch = arr_patch[sy:ey, sx:ex]

            red = patch[idx_C > 0]
            red.sort()
            median = int(np.median(red))

            if np.abs(red[-1] - red[-2]) > 128:

                print('-->', yy, xx)

                # idx = patch[(patch == red[-1]) & (idx_C > 0)]
                # print(idx)
                # patch[idx] = median
                patch[(patch == red[-1]) & (idx_C > 0)] = median

                print('hello dynamic bp')
                # exit()

            arr_patch[sy:ey, sx:ex] = patch

    return arr_patch



def main():

    if False:
        fname = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val/0001.png'
        image = PIL.Image.open(fname)

        image_arr = np.array(image)

        print(image_arr.shape, image_arr.dtype)

        # crop
        offset = 100
        sample_orig = image_arr[offset:offset+128, offset:offset+128]
        sample_noise = np.copy(sample_orig)


        # add salt noise
        number_of_pixels = 200
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, 128 - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, 128 - 1)

            # Color that pixel to white
            sample_noise[y_coord][x_coord] = 255

        # make tetra
        sample_tetra_noise = sample_noise * idx_RGB.astype(np.uint8)
        sample_tetra_noise_1ch = np.sum(sample_tetra_noise, axis=-1)

        sample_tetra_cure_1ch = cure_dynamic_bp(sample_tetra_noise_1ch)
        sample_tetra_cure_R = sample_tetra_cure_1ch * idx_R
        sample_tetra_cure_G = sample_tetra_cure_1ch * idx_G
        sample_tetra_cure_B = sample_tetra_cure_1ch * idx_B

        sample_tetra_cure = np.stack((sample_tetra_cure_R, sample_tetra_cure_G, sample_tetra_cure_B), axis=-1)
        sample_tetra_cure = sample_tetra_cure.astype(np.uint8)

        # print(sample_tetra_noise[:4, :4])

        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(sample_orig)

        plt.subplot(2, 2, 2)
        plt.imshow(sample_noise)

        plt.subplot(2, 2, 3)
        plt.imshow(sample_tetra_noise)

        plt.subplot(2, 2, 4)
        plt.imshow(sample_tetra_cure)

        plt.show()



    else:
        fname_in = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val/0001_0080_in.npy'
        fname_gt = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val/0001_0080_gt.npy'
        sample_tetra_noise_1ch = np.load(fname_in)
        sample_tetra_noise_R = sample_tetra_noise_1ch * idx_R
        sample_tetra_noise_G = sample_tetra_noise_1ch * idx_G
        sample_tetra_noise_B = sample_tetra_noise_1ch * idx_B
        sample_tetra_noise = np.stack((sample_tetra_noise_R, sample_tetra_noise_G, sample_tetra_noise_B), axis=-1)/2
        sample_tetra_noise = sample_tetra_noise.astype(np.uint8)

        sample_tetra_gt = np.load(fname_gt)

        sample_tetra_cure_1ch = cure_dynamic_bp(sample_tetra_noise_1ch)
        sample_tetra_cure_R = sample_tetra_cure_1ch * idx_R
        sample_tetra_cure_G = sample_tetra_cure_1ch * idx_G
        sample_tetra_cure_B = sample_tetra_cure_1ch * idx_B

        sample_tetra_cure = np.stack((sample_tetra_cure_R, sample_tetra_cure_G, sample_tetra_cure_B), axis=-1)/2
        sample_tetra_cure = sample_tetra_cure.astype(np.uint8)

        print(np.amin(sample_tetra_noise), np.amax(sample_tetra_noise))
        print(np.amin(sample_tetra_cure), np.amax(sample_tetra_cure))

        # print(sample_tetra_noise[:4, :4])


        plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(sample_tetra_gt)

        plt.subplot(1,3,2)
        plt.imshow(sample_tetra_noise)

        plt.subplot(1,3,3)
        plt.imshow(sample_tetra_cure)

        plt.show()




if __name__ =='__main__':
    main()