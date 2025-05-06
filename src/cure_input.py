import os
import glob
import numpy as np
import matplotlib.pyplot as plt


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



def cure_dynamic_bp(arr_patch):
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O
    #   R R G G R R G G  |  O O X X O O X X
    #   R R G G R R G G  |  O O X X O O X X
    #   G G B B G G B B  |  X X O O X X O O
    #   G G B B G G B B  |  X X O O X X O O

    cfa_pattern = 2
    crop_size = 8

    idx_R = np.tile(
        np.concatenate(
            (np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                    np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)),
            axis=0),
        (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))




    # print('-->', idx_R>0)
    for yy in range(0, arr_patch.shape[0]-8, 2):
        for xx in range(0, arr_patch.shape[1]-8, 2):
            sy = yy
            ey = sy+8
            sx = xx
            ex = sx+8
            patch = arr_patch[sy:ey, sx:ex]

            red = patch[idx_R > 0]
            red.sort()
            median = int(np.median(red))

            mymax = 128
            threshold = (red[-1] / 1023 ) * mymax

            if np.abs(red[-1] - red[-2]) > threshold:
                # idx = patch[(patch == red[-1]) & (idx_R == 1)]
                # patch[idx] = median
                patch[(patch == red[-1]) & (idx_R > 0)] = median

                # print('hello dynamic bp')
                # exit()

            arr_patch[sy:ey, sx:ex] = patch

    return arr_patch





def main():

    path = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val'
    npyfiles = glob.glob(os.path.join(path, 'input/*.npy'))
    npyfiles.sort()
    print(len(npyfiles), npyfiles)

    for idx, file in enumerate(npyfiles):
        print(idx, file)
        arr_patch = np.load(file)    # (0, 1023)

        # cure static bp
        arr_patch = cure_static_bp(arr_patch)

        # cure dynamic bp
        arr_patch = cure_dynamic_bp(arr_patch)

        newfname = 'input_cure/%04d.npy' % (idx + 1 + 800)
        newfname = os.path.join(path, newfname)
        np.save(newfname, arr_patch)




if __name__ == '__main__':
    main()
