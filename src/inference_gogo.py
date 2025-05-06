import os
import numpy as np
import time
import h5py
import tensorflow as tf
import skimage.io
import glob
import argparse


# tf.compat.v1.disable_eager_execution()

def get_input(raw, input_type, cell_size):


    idx_R = np.tile(np.concatenate(
            (np.concatenate((np.zeros((cell_size, cell_size)), np.ones((cell_size, cell_size))), axis=1),
             np.concatenate((np.zeros((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1)), axis=0),
            (raw.shape[1] // cell_size // 2, raw.shape[2] // cell_size // 2)).astype(bool)

    idx_G = np.tile(np.concatenate(
            (np.concatenate((np.ones((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1),
             np.concatenate((np.zeros((cell_size, cell_size)), np.ones((cell_size, cell_size))), axis=1)), axis=0),
            (raw.shape[1] // cell_size // 2, raw.shape[2] // cell_size // 2)).astype(bool)

    idx_G1 = np.tile(np.concatenate(
            (np.concatenate((np.ones((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1),
             np.concatenate((np.zeros((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1)), axis=0),
            (raw.shape[1] // cell_size // 2, raw.shape[2] // cell_size // 2)).astype(bool)

    idx_G2 = np.tile(np.concatenate(
            (np.concatenate((np.zeros((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1),
             np.concatenate((np.zeros((cell_size, cell_size)), np.ones((cell_size, cell_size))), axis=1)), axis=0),
            (raw.shape[1] // cell_size // 2, raw.shape[2] // cell_size // 2)).astype(bool)

    idx_B = np.tile(np.concatenate(
            (np.concatenate((np.zeros((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1),
             np.concatenate((np.ones((cell_size, cell_size)), np.zeros((cell_size, cell_size))), axis=1)), axis=0),
            (raw.shape[1] // cell_size // 2, raw.shape[2] // cell_size // 2)).astype(bool)

    idx_RGB = np.concatenate((idx_R[..., np.newaxis],
                              idx_G[..., np.newaxis],
                              idx_B[..., np.newaxis]), axis=-1)


    raw[0, ..., 0][np.logical_not(idx_R)] = 0
    raw[0, ..., 1][np.logical_not(idx_G)] = 0
    raw[0, ..., 2][np.logical_not(idx_B)] = 0

    if input_type in ['nonshrink']:
        return raw


    if input_type in ['nonshrink_4ch']:

        g1 = np.multiply(raw[..., 1], idx_G1)
        r  = np.multiply(raw[..., 0], idx_R)
        b  = np.multiply(raw[..., 2], idx_B)
        g2 = np.multiply(raw[..., 1], idx_G2)
        # print(idx_G1.shape, g1.shape)
        # exit()

        patternized = np.concatenate((g1[..., np.newaxis],
                                      r[..., np.newaxis],
                                      b[..., np.newaxis],
                                      g2[..., np.newaxis]),
                                     axis=-1)


        return patternized



    if input_type in ['shrink']:
        print('shape idx rgb', idx_G1.shape, idx_G2.shape, idx_R.shape, idx_B.shape)
        pat_g1 = raw[0, ..., 1][idx_G1]
        print('shape pat_g1b', pat_g1.shape)

        h2 = raw.shape[1] // 2
        w2 = raw.shape[2] // 2
        print('shrink hxw', raw.shape, h2, w2)

        patternized = np.zeros((1, h2, w2, 4), dtype=np.uint8)
        print('patt ', patternized.shape, patternized[0, ..., 0].shape)
        # G1
        patternized[0, ..., 0] = raw[0, ..., 1][idx_G1].reshape(h2, w2)
        # R
        patternized[0, ..., 1] = raw[0, ..., 0][idx_R].reshape(h2, w2)
        # B
        patternized[0, ..., 2] = raw[0, ..., 2][idx_B].reshape(h2, w2)
        # G2
        patternized[0, ..., 3] = raw[0, ..., 1][idx_G2].reshape(h2, w2)

        print('patt ', patternized.shape, patternized[0, ..., 0].shape)

        return patternized

    raise ValueError('unknown input type', input_type)

    return None

def get_image_left_right_average(RB):
    left = RB[:, 0::2]
    right = RB[:, 1::2]
    avg = (left + right) / 2.
    superpd = np.repeat(avg, 2, axis=1)
    return superpd
#





def main(args):


    ## get argument
    input_type = args.input_type
    cell_size = args.cell_size
    rb_option = args.rb_option
    input_max =  args.input_max

    dic_args = vars(args)
    for s in list(dic_args.keys()):
        print('args.', s, ': ', dic_args[s])

    if input_type not in ['nonshrink_4ch']:
        raise ValueError('unknown input type', input_type)


    base_dir = 'model_dir_bwunet_sedecNone_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_sedecRBSuper_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'

    base_dir = 'model_dir_bwunet_tetraNone_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraNone_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraRBSuper_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraRBSuper_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'



    ## RBSuper
    base_dir = 'model_dir_bwunet_tetraRBSuper0001_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraRBSuper0001_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'

    base_dir = 'model_dir_bwunet_tetraRBSuper0256_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraRBSuper0256_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'

    base_dir = 'model_dir_bwunet_tetraRBSuper1024_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraRBSuper1024_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'


    # Normal
    base_dir = 'model_dir_bwunet_tetraNone0001_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraNone0001_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'

    base_dir = 'model_dir_bwunet_tetraNone0256_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraNone0256_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'

    base_dir = 'model_dir_bwunet_tetraNone1024_prelu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'
    base_dir = 'model_dir_bwunet_tetraNone1024_relu_3dsr_ed2_64x64_16_32_64_128_0001_const6.0'


    # # Qcell
    base_dir = 'model_dir_bwunet_tetraQCell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6_c2c'
    base_dir = 'model_dir_bwunet_tetraQCell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6_n2n'


    ## Activation Test
    base_dir = "model_dir_bwunet_tetraNone_3dsr_ed2_64x64_16_32_64_128_0001_nonshrink_4ch_const6.0"
    base_dir = "model_dir_bwunet_tetraNone_3dsr_ed2_64x64_16_32_64_128_0001_nonshrink_4ch_const6.0_at"
    base_dir = "model_dir_bwunet_tetraNone_3dsr_ed2_64x64_16_32_64_128_1111_nonshrink_4ch_const6.0"
    base_dir = "model_dir_bwunet_tetraNone_3dsr_ed2_64x64_16_32_64_128_1111_nonshrink_4ch_const6.0_at"

    # TetraQcell
    # # base_dir = 'model_dir_bwunet_tetraQCell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6_c2c'
    # # base_dir = 'model_dir_bwunet_tetraQCell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6


    # Tetra2Qcell
    base_dir = 'model_dir_bwunet_tetra2qcell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6_lc'
    base_dir = 'model_dir_bwunet_tetra2qcell_3dsr_ed2_upscale_32x32_32_64_128_256_0001_nonshrink_4ch_const6_hc'

    # TetraQcell
    base_dir = 'model_dir_bwunet_tetraqcell_3dsr_ed2_upscale_32x32_16_32_64_128_0001_nonshrink_4ch_const6_lc'
    base_dir = 'model_dir_bwunet_tetraqcell_3dsr_ed2_upscale_32x32_32_64_128_256_0001_nonshrink_4ch_const6_hc'


    # model dir
    base_dir = 'model_dir_bwunet_tetra2qcell_3dsr_ed2_upscale_delta_32x32_32_64_128_256_0001_nonshrink_4ch_const6.0_hc_clean'
    base_dir = 'model_dir_bwunet_tetra2qcell_3dsr_ed2_upscale_delta_32x32_32_64_128_256_0001_nonshrink_4ch_const6.0_hc_noise2noise'
    base_dir = 'model_dir_bwunet_tetraqcell_3dsr_ed2_upscale_delta_32x32_32_64_128_256_0001_nonshrink_4ch_const6.0_hc_clean'
    base_dir = 'model_dir_bwunet_tetraqcell_3dsr_ed2_upscale_delta_32x32_32_64_128_256_0001_nonshrink_4ch_const6.0_hc_noise2noise'
    # model dir

    # Tetra2 binning
    base_dir = 'model_dir_bwunet_3dsr_ed3_tetra2_binning_wrapper_delta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_tetra2_binning_wrapper_nondelta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_tetra2_full_wrapper_delta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_tetra2_full_wrapper_nondelta_0001_64x64_ech_16_32_64_128_dch_64_32_16'


    ## bw geo
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_binning_wrapper_delta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_binning_wrapper_delta_0001_64x64_ech_32_64_128_256_dch_128_64_32'
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_binning_wrapper_nondelta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    # base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_binning_wrapper_nondelta_0001_64x64_ech_32_64_128_256_dch_128_64_32'


    ## tetra
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra_full_wrapper_delta_0001_64x64_ech_32_64_128_256_dch_128_64_32'
    # base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra_full_wrapper_nondelta_0001_64x64_ech_32_64_128_256_dch_128_64_32'


    ## tetra2
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_full_wrapper_delta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_full_wrapper_delta_0001_64x64_ech_32_64_128_256_dch_128_64_32'
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_full_wrapper_nondelta_0001_64x64_ech_16_32_64_128_dch_64_32_16'
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra2_full_wrapper_nondelta_0001_64x64_ech_32_64_128_256_dch_128_64_32'



    # binning stack
    base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra_binning_stack_wrapper_delta_0001_64x64_ech_16_32_64_128_dch_64_32_16_bw_test'
    # base_dir = 'model_dir_bwunet_3dsr_ed3_geo_tetra_binning_stack_wrapper_delta_0001_64x64_ech_32_64_128_256_dch_128_64_32_bw_test'


    # use dir in ckpts

    # base_dir = os.path.join('ckpts', base_dir)

    # if  "None0001_" in base_dir:
    #     input_max = 1.
    # elif "0256_" in base_dir:
    #     input_max = 255.
    # elif "1024_" in base_dir:
    #     input_max = 1023.

    # input_max = 1.

    print('----------- input_max = ', input_max)



    if '_prelu_' in base_dir:
        activation = 'prelu'
    else:
        activation = 'relu'

    if '_tetra_' in base_dir.lower():
        cfa_pattern = 'Tetra'
        cell_size = 2


    elif '_tetra2_' in base_dir.lower():
        cfa_pattern = 'Tetra2'
        cell_size = 4
    else:
        ValueError('unknown cfa')

    # print("cfa_pattern, ", cfa_pattern)
    # exit()

    # get model from yaml

    try:
        json_file = glob.glob(os.path.join(base_dir, '*.json'))
        json_file = json_file[0]
        print(json_file)
        json_string = open(json_file).read()
        model = tf.keras.models.model_from_json(json_string)

        # h5file = glob.glob(os.path.join(base_dir, '*.h5'))
        # h5file = h5file[0]
        # print(h5file)
        # model = tf.kreas.models.load_model(h5file, custom_objects={'tf':tf})

    except Exception:
        # get model from json
        try:
            yaml_file = glob.glob(base_dir + '/*.yaml')
            yaml_file = yaml_file[0]
            print(yaml_file)
            yaml_string = open(yaml_file).read()
            model = tf.keras.models.model_from_yaml(yaml_string)

            # h5file = glob.glob(base_dir + '/models/*.h5')
            # h5file = h5file[0]
            # print(h5file)
            # model = tf.kreas.models.load_model(h5file, custom_objects={'tf':tf})


        except Exception:


            # get model from json
            try:
                h5_files = glob.glob(base_dir + '/*.h5')
                h5_files = h5_files[0]
                print(h5_files)
                model = tf.keras.models.load_model(h5_files, custom_objects={'tf':tf})


                # h5file = glob.glob(base_dir + '/models/*.h5')
                # h5file = h5file[0]
                # print(h5file)
                # model = tf.kreas.models.load_model(h5file)
            except Exception:

                print('neither h5, json nor yaml exists!')
                exit()

    # get latest weight
    weight = glob.glob(base_dir + '/models/*')
    weight.sort()
    weight = weight[-1]
    print(weight)
    epoch = os.path.basename(weight).split('_')[0]
    print(epoch)

    # epoch=000000
    # weight = os.path.join('model_dir_bwunet_3dsr_ed_64x64_16_32_64_128_last1_no_pool_ploss',
    #                       'ori_bwunet_3dsr_16_32_64_128_last1_maxpooling_no_diff_1583996011.h5')
    # weight = os.path.join('model_dir_bwunet_3ds_ed_128x128_16_32_64_128_last2_ploss',
    #                       'ed_12_32_0010.h5')


    # model = tf.keras.models.load_model(weight)
    model.load_weights(weight)
    model.summary()


    # output dir
    output_dir = os.path.join(base_dir, 'inference')
    os.makedirs(output_dir, exist_ok=True)

    f = h5py.File('kodak_data_only.h5', 'r')
    files = enumerate(f)
    keys = enumerate(f.keys())

    print('----------- input_max = ', input_max)
    print('----------- cell_size = ', cell_size)



    psnr = 0
    ssim = 0
    for idx, key in keys:

        ## get raw
        raw = f[key]

        raw_orig = np.copy(raw[0])
        # print('raw.shape', raw.shape, raw.dtype, type(raw))
        # print('amin, amax', np.amin(raw), np.amax(raw))

        ## dimension
        h, w = raw.shape[1:3]


        PAD_SIZE_FULL = 128  # 64#
        PAD_NUM = 8
        PAD_SIZE = PAD_NUM * 2
        psize = PAD_SIZE_FULL - PAD_SIZE

        PAD_F = PAD_NUM
        PAD_H = PAD_SIZE_FULL - (PAD_F + h) % psize
        PAD_W = PAD_SIZE_FULL - (PAD_F + w) % psize

        # print('patternized.shape', patternized.shape)
        # print('========> patternized', patternized.shape)

        # raw = np.expand_dims(raw, axis=0)

        # get patternized input
        # print(idx, key)
        # if inference from known RGB image
        p2d = ((0, 0), (PAD_F, PAD_H), (PAD_F, PAD_W), (0, 0))
        raw = np.pad(raw, p2d, 'reflect')
        raw = raw.astype(np.float32)



        # raw = get_input(raw, input_type, cell_size)


        if rb_option in ['RBSuper']:
            raw[0, ..., 0] = get_image_left_right_average(raw[0, ..., 0]) # R
            raw[0, ..., 2] = get_image_left_right_average(raw[0, ..., 2]) # B



        # print('========> raw', raw.shape)

        # lets get it done
        img_dmsc = np.zeros((1, h, w, 3))
        rangex = range(0, h, psize)
        rangey = range(0, w, psize)


        if rb_option in ['RBSuper']:
            print('RBSuper')
        for start_x in rangex:
            for start_y in rangey:
                end_x = start_x + psize
                end_y = start_y + psize


                tileRaw = raw[:, start_x:end_x + PAD_SIZE, start_y:end_y + PAD_SIZE, :].astype(np.float32)

                tileRaw = get_input(tileRaw, input_type, cell_size)


                tileRaw = (tileRaw / 255.)
                tileRaw = (tileRaw * input_max)

                # print('amin, amax', np.amin(tileRaw), np.amax(tileRaw))

                tileout = model.predict(tileRaw)
                # bias = 0.5 / 255.
                # tileout += bias

                # tileout = tileout / 255.
                tileout = tileout / input_max

                end_x_pad = PAD_SIZE_FULL - PAD_NUM
                end_y_pad = PAD_SIZE_FULL - PAD_NUM

                if end_x > h:
                    end_x = h
                    end_x_pad = PAD_NUM + h % psize

                if end_y > w:
                    end_y = w
                    end_y_pad = PAD_NUM + w % psize

                img_dmsc[:, start_x:end_x, start_y:end_y, :] = tileout[:, PAD_NUM:end_x_pad, PAD_NUM:end_y_pad, :]



                # print('tile size:',start_x,end_x+PAD_SIZE,start_y,end_y+ PAD_SIZE)
                # print('pad size:',start_x,end_x,start_y,end_y)



        img_cliped = np.clip(img_dmsc[0, ...], 0, 1)

        img_name = key + '_' + '%s' % epoch + '_%s.png' % input_type
        img_name = '%02d_%s_%s%s_%s_%s.png' % (idx + 1, input_type, cfa_pattern, rb_option, activation, epoch)

        file_output_path = os.path.join(output_dir, img_name)
        skimage.io.imsave(file_output_path, img_cliped)

        ## psnr / ssim
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity



        img_dmsc_clip = np.clip(img_dmsc[0, ...], 0, 1)

        if idx < 24:
            img_dmsc_clip = img_dmsc_clip * 255
            img_dmsc_clip = img_dmsc_clip.astype(np.int)
            p = peak_signal_noise_ratio(raw_orig, img_dmsc_clip, data_range=255.)
            s = structural_similarity(raw_orig, img_dmsc_clip, data_range=255., multichannel=True)
            psnr += p
            ssim += s
            print(idx, ' in, psnr %.3fdB' % p)
        else:
            print(idx, ' out')

        # print(idx, 'done')

    f.close()
    print('PSNR', psnr / 24, ', SSIM', ssim / 24)
    psnr_ssim = 'PSNR: %.3f, SSIM: %.3f' % (psnr / 24, ssim / 24)

    psnr_ssim = os.path.join(output_dir, psnr_ssim)
    os.makedirs(psnr_ssim, exist_ok=True)
    print('--------------', psnr_ssim)

    pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_type',
            type=str,
            default='nonshrink_4ch',
            help='shrink / shrinkRB/ nonshrink / nonshrink_4ch. default:shrink')

    parser.add_argument(
            '--cell_size',
            type=int,
            default=2,
            help='2/3/4. default:2')

    parser.add_argument(
            '--input_max',
            type=float,
            # default=255,
            default=1.,
            help='1/255/1023. default:1')
    parser.add_argument(
            '--inference_type',
            type=str,
            # default=255,
            default='with_wrapper', # 'delta_only', 'with_wrapper'
            help='delta_only/with_wrapper,  default:delta_only')
    parser.add_argument(
            '--sedec_option',
            type=str,
            default='None',
            help='None, "RBSuper". default:None')
    parser.add_argument(
            '--rb_option',
            type=str,
            # default='RBSuper',
            # default=None,
            help='None, "RBSuper". default:RBSuper')

    args = parser.parse_args()

    main(args)

    print('done')
