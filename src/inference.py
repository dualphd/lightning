import os, glob, math
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from myutils_tf import bwutils


cfa_pattern = 2
crop_size = 8

idx_R = np.tile(
        np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                              np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                        (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))


def get_model(model_name, model_sig):
    base_path = os.path.join('model_dir', 'checkpoint')
    structure_path = os.path.join(base_path, model_name + '_model_structure.h5')
    ckpt_path = os.path.join(base_path, model_name + '_' + model_sig)
    print(structure_path, '\n', ckpt_path)


    # load model structure
    model = tf.keras.models.load_model(structure_path)

    # find latest weights and load
    ckpts = glob.glob(os.path.join(ckpt_path, '*.h5'))
    ckpts.sort()
    ckpt = ckpts[-1]
    model.load_weights(ckpt)

    print(ckpt)
    # model.summary()
    return model


def cure_static_bp(arr_patch):
    ## Red
    for yy in range(1, arr_patch.shape[0], 4):
        for xx in range(1, arr_patch.shape[1], 4):
            arr_patch[yy][xx] = ((arr_patch[yy - 1][xx] + arr_patch[yy][xx - 1]) / 2)

    ## Blue
    for yy in range(3, arr_patch.shape[0], 4):
        for xx in range(3, arr_patch.shape[1], 4):
            arr_patch[yy][xx] = ((arr_patch[yy - 1][xx] + arr_patch[yy][xx - 1]) / 2)
    return arr_patch


def cure_dynamic_bp(arr_patch, nbp=4):
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O


    # print('-->', idx_R>0)
    for yy in range(0, arr_patch.shape[0]-8, 2):
        for xx in range(0, arr_patch.shape[1]-8, 2):
            sy = yy
            ey = sy + 8
            sx = xx
            ex = sx + 8
            patch = arr_patch[sy:ey, sx:ex]

            for _ in range(nbp):
                red = patch[idx_R > 0]
                red.sort()
                median = int(np.median(red))
                if np.abs(red[-1] - red[-2]) > (64 / 1023):
                    patch[(patch == red[-1]) & (idx_R > 0)] = median

            arr_patch[sy:ey, sx:ex] = patch

    return arr_patch




def main(model_name, model_sig):
    # model name

    # get model
    model = get_model(model_name, model_sig)
    model.summary()

    # exit()


    # cellsize
    cell_size=2
    cfa_pattern = 'tetra'





    # test data
    PATH_VAL = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val/input_cure'
    files = glob.glob(os.path.join(PATH_VAL, '*.npy'))
    files.sort()
    pad_size = 8
    # pad_size = 0
    patch_size = 128


    # utils for patternized
    utils = bwutils(input_type='nonshrink',
                    cfa_pattern=cfa_pattern,
                    patch_size=patch_size,
                    crop_size=patch_size,
                    input_max=255,
                    use_unprocess=False,
                    loss_type=['rgb'],
                    loss_mode='2norm',
                    loss_scale=1e4,
                    cache_enable=False)


    # exit()

    # shape = np.load(files[0]).shape
    # height, width, channels = np.load(files[0]).shape
    # npatches_y, npatches_x = math.ceil(shape[0]/patch_size), math.ceil(shape[1]/patch_size)
    # print(arr_pred.shape)
    for idx, file in enumerate(files):
        arr = np.load(file)    # (0, 65535)
        arr = arr / (2**10 -1) # (0, 1)
        # arr = arr * 2 -1
        # arr = arr ** (1/2.2)   # (0, 1)

        # print('min, max, ', np.amin(arr), np.amax(arr))
        # exit()

        ## padding

        # h, w = arr.shape
        # PAD_SIZE_FULL = 128  # 64#

        # pad width
        arr = np.concatenate([arr[:, cell_size + pad_size:cell_size:-1],
                              arr,
                              arr[:, -cell_size - 1: -cell_size - pad_size - 1:-1]], axis=1)

        # pad height
        arr = np.concatenate([arr[cell_size + pad_size:cell_size:-1, :],
                              arr,
                              arr[-cell_size - 1: -cell_size - pad_size - 1:-1, :]], axis=0)
        print('arr.shape', arr.shape)


        # arr = np.fliplr(arr)
        # arr = np.flipud(arr)



        assert arr.shape[0]%4 == 0 and arr.shape[1]%4 == 0, f'{idx}, arr shape not in multiple of 4'
        # continue
        # break

        height, width = arr.shape
        npatches_y = math.ceil((height+2*pad_size) / (patch_size-2*pad_size))
        npatches_x = math.ceil((width +2*pad_size) / (patch_size-2*pad_size))

        # ..................................................................................
        # red = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # green = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
        # blue = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
        #
        # idx_red   = np.tile(red,   (height // cell_size // 2, width // cell_size // 2))
        # idx_green = np.tile(green, (height // cell_size // 2, width // cell_size // 2))
        # idx_blue  = np.tile(blue,  (height // cell_size // 2, width // cell_size // 2))
        #
        # Red   = arr * idx_red
        # Green = arr * idx_green
        # Blue  = arr * idx_blue
        #
        # RGB = np.concatenate([Red[...,np.newaxis], Green[...,np.newaxis], Blue[...,np.newaxis]], axis=-1)
        #
        # img_tetra = Image.fromarray((RGB * 255 + 0.5).astype(np.uint8))
        # # name = os.path.join(PATH_PIXELSHIFT, f'inf_{model_name}_{model_sig}_%02d.png'%(idx+1))
        # name = os.path.join(PATH_VAL, f'%04d_tetra.png' % (idx + 801))
        # img_tetra.save(name)
        # exit()
        # ..................................................................................








        # arr_pred = np.zeros_like(arr)
        # arr_pred = arr_pred[...,np.newaxis]
        arr_pred = np.zeros(arr.shape + (3,) )
        print(idx, file, arr.shape, arr_pred.shape)
        # exit()
        cnt=0
        tcnt= npatches_x*npatches_y
        for idx_y in range(npatches_y):
            for idx_x  in range(npatches_x):
        # for idx_y in range(npatches_y, 0, -1):
        #     for idx_x  in range(npatches_x, 0, -1):
                if(cnt%10==0):
                    print(f'{idx} : {cnt} / {tcnt}')
                cnt+=1
                sy = idx_y * (patch_size-2*pad_size)
                ey = sy + patch_size
                sx = idx_x * (patch_size-2*pad_size)
                ex = sx + patch_size



                if ey >= height:
                    ey = height
                    sy = height - patch_size

                if ex >= width:
                    ex = width
                    sx = width - patch_size


                arr_patch = arr[sy:ey, sx:ex]

                ####################################################################################
                ####################################################################################
                # # cure static bp
                # arr_patch = cure_static_bp(arr_patch)
                #
                # # cure dynamic bp
                # arr_patch = cure_dynamic_bp(arr_patch)
                ####################################################################################
                ####################################################################################


                arr_patch = utils.get_patternized_1ch_to_3ch_image(arr_patch)



                arr_patch = (arr_patch*2) - 1  # (0, 1) -> (-1, 1)

                # prediction
                pred = model.predict(arr_patch[np.newaxis,...], verbose=0)



                # post-process
                if pad_size == 0:
                    arr_pred[sy:ey, sx:ex, :] = pred[0]
                else:
                    pass
                    arr_pred[sy+pad_size:ey-pad_size, sx+pad_size:ex-pad_size, :] = \
                                pred[0, pad_size:-pad_size, pad_size:-pad_size, :]



                # exit()
        # exit()

        if pad_size > 0:
            arr_pred = arr_pred[pad_size:-pad_size, pad_size:-pad_size, :]
        arr_pred = (arr_pred+1) / 2  # normalized from (-1, 1) to (0,1)
        arr_pred = ((arr_pred*255) + 0.5).astype(np.uint8)
        print(arr_pred.shape, np.amin(arr_pred), np.amax(arr_pred))
        img_pred = Image.fromarray(arr_pred)
        name = os.path.join(PATH_VAL, f'%04d.png'%(idx+801))
        img_pred.save(name)
        # exit()

def run():

    args = [
            {'model_name':'bwunet', 'model_sig':'noise'}
            # {'model_name': 'bwunet', 'model_sig': 'single'}
            # {'model_name':'bwunet_delta', 'model_sig':'noise'}
            ]
    for arg in args:
        model_name = arg['model_name']
        model_sig  = arg['model_sig']
        main(model_name, model_sig)

if __name__ == '__main__':
    run()
