

import os
import random
import glob
import tensorflow as tf
import numpy as np
from functools import partial

from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
from collections import OrderedDict

TETRA = 2

TF_VER=1 if tf.__version__.split('.')[0]=='1' else (2 if tf.__version__.split('.')[0]=='2' else None)



def load_checkpoint_if_exists(model, model_dir, model_name, ckpt_name=None):
    prev_epoch = 0

    prev_loss = np.inf
    trained_model_file_name = os.path.join(model_dir, ckpt_name)
    if (ckpt_name != None ) and \
       (os.path.isfile(trained_model_file_name) and os.path.exists(trained_model_file_name)):

        print('=============> load designated trained model: ', ckpt_name)
        model.load_weights(trained_model_file_name)
        idx = trained_weights.rfind(model_name)
        prev_epoch = int(trained_weights[idx-6:idx-1])
        prev_loss = float(trained_weights.split('_')[-1][:-3])

        # raise ValueError('unkown trained weight', trained_model_file_name)
    else:
        if (trained_model_file_name != None ) and (not os.path.exists(trained_model_file_name)):
            print('=============> pre trained designated model not exists ')

        trained_weights = glob.glob(os.path.join(model_dir, '*.h5' ))
        print('--=-=-=-> ', trained_weights)
        if len(trained_weights ) > 0:
            print('===========> %d TRAINED WEIGHTS EXIST' % len(trained_weights))
            trained_weights.sort()
            trained_weights = trained_weights[-1]
            print('---------------------> ', trained_weights)
            model.load_weights(trained_weights)
            # idx = trained_weights.rfind(model_name)
            # prev_epoch = int(trained_weights[idx-6:idx-1])
            # prev_loss = float(trained_weights.split('_')[-1][:-3])
            wname = trained_weights.split('/')[-1]
            prev_epoch = int(wname[:5])
            print('prev epoch', prev_epoch)
        else:
            print('===========> TRAINED WEIGHTS NOT EXIST', len(trained_weights))

    return model, prev_epoch, prev_loss




class bwutils():

    def __init__(self,
                input_type='nonshrink',
                output_type='data_only', # 'data_only', 'data_with_mask'
                cfa_pattern='tetra',
                file_type='tfrecord',
                patch_size:int=128,
                crop_size:int=128,
                batch_size:int=128,
                lattice_fsize:int=3,
                lattice_factor:float=1e-5,
                input_max = 1.,
                input_bits =16,
                input_bias = True,
                use_unprocess=False,
                loss_scale = 1,
                alpha_for_gamma = 0.05,
                beta_for_gamma = (1./2.2),
                upscaling_factor=None,
                upscaling_method='bilinear',
                loss_type=['rgb', 'yuv'], # 'rgb', 'yuv', 'ploss', 'ssim'.
                loss_weight={'rgb':1, 'yuv':1 },
                loss_mode='2norm',
                cache_enable=False):

        if input_type not in ['shrink', 'nonshrink', 'nonshrink_4ch', 'shrink_upscale', 'raw_1ch', 'rgb']:
            raise ValueError('unknown input_type  '
                             'input type must be either "shrink" / "nonshrink" / "nonshrink_4ch" / "raw_1ch"  but', input_type)

        cfa_pattern = cfa_pattern.lower()
        if cfa_pattern in ['tetra', 2]:
            cfa_pattern = 2
        else:
            raise ValueError('unknown cfa_pattern, ', cfa_pattern)

        for lt  in loss_type:
            if lt not in ['rgb', 'yuv', 'ploss', 'ssim', 'dct', 'lattice', 'dir']:
                raise ValueError('unknown loss type, ', lt)

        self.cfa_pattern = cfa_pattern
        self.file_type = file_type
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.input_bits = input_bits
        self.input_max = input_max
        self.input_bias = input_bias
        self.use_unprocess = use_unprocess
        self.alpha_for_gamma = alpha_for_gamma
        self.beta_for_gamma =  beta_for_gamma
        self.upscaling_factor = upscaling_factor
        self.upscaling_method = upscaling_method
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.loss_scale = loss_scale
        self.cache_enable = cache_enable
        self.lattice_factor=lattice_factor

        self.input_type = input_type
        self.output_type = output_type


        if input_bits == 8:
            self.dtype='uint8'
        elif input_bits == 16:
            self.dtype='uint16'
        else:
            print('known input_bits', input_bits)
            exit()

        # self.input_scale = input_max / 255.

        if loss_mode == 'square' or loss_mode == 'mse' or loss_mode=='2norm':
            self.loss_norm = tf.keras.backend.square
        elif loss_mode == 'abs' or loss_mode=='1norm':
            self.loss_norm = tf.keras.backend.abs
        else:
            ValueError('unknown loss_mode %s' %  loss_mode)


        self.crop_margin = int(patch_size - crop_size)


        self.idx_R = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G1 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G2 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_B = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        self.idx_G = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))


        self.idx_RGB = np.concatenate((self.idx_R[..., np.newaxis],
                                       self.idx_G[..., np.newaxis],
                                       self.idx_B[..., np.newaxis]), axis=-1)

        self.idx_G1RBG2 = np.concatenate((self.idx_G1[..., np.newaxis],
                                          self.idx_R[..., np.newaxis],
                                          self.idx_B[..., np.newaxis],
                                          self.idx_G2[..., np.newaxis]), axis=-1)

        # lattice loss
        idx_fft=np.zeros((batch_size, 3, crop_size, crop_size//2+1), dtype=np.bool_)
        period=cfa_pattern*2
        for r in range(-(lattice_fsize//2), (lattice_fsize//2)+1):
            for c in range(-(lattice_fsize//2), (lattice_fsize//2)+1):
                if r<0: rpos=r+crop_size//period
                else: rpos=r
                if c<0: cpos=c+crop_size//period
                else: cpos=c
                idx_fft[:,:,rpos:crop_size//2+1:crop_size//period,cpos:crop_size//2+1:crop_size//period]=1
        idx_fft[:,:,:(lattice_fsize)//2+1,:(lattice_fsize//2)+1]=0
        self.tf_idx_fft=tf.cast(idx_fft, dtype=tf.bool)

        # dir loss
        filter_H_dir = np.zeros((5, 5,1,15), dtype=np.float32)
        filter_V_dir = np.zeros((5, 5,1,15), dtype=np.float32)
        filter_S_dir = np.zeros((5, 5,1,13), dtype=np.float32)
        filter_B_dir = np.zeros((5, 5,1,13), dtype=np.float32)
        filter_P_dir = np.zeros((5, 5,1,13), dtype=np.float32)

        for r in range(1,4):
            for c in range(5):
                filter_H_dir[r,c,0,(r-1)*5+c]=1
                filter_V_dir[c,r,0,(r-1)*5+c]=1

        for r in range(5):
            filter_S_dir[r,4-r,0,r]=1
            filter_B_dir[r,r,0,r]=1
            if r==4: continue
            filter_S_dir[r,3-r,0,r+5]=1
            filter_S_dir[r+1, 4 - r, 0, r + 9] = 1
            filter_B_dir[r,r+1,0,r+5]=1
            filter_B_dir[r+1,r,0,r+9]=1
        cnt=0
        for r in range(5):
            for c in range(5):
                if np.abs(r-2)+np.abs(c-2)>2: continue
                filter_P_dir[r,c,0,cnt]=1
                cnt+=1
        self.tf_filter_H_dir = tf.cast(filter_H_dir, dtype=tf.float32)
        self.tf_filter_V_dir = tf.cast(filter_V_dir, dtype=tf.float32)
        self.tf_filter_S_dir = tf.cast(filter_S_dir, dtype=tf.float32)
        self.tf_filter_B_dir = tf.cast(filter_B_dir, dtype=tf.float32)
        self.tf_filter_P_dir = tf.cast(filter_P_dir, dtype=tf.float32)

        print('[bwutils] input_type', input_type)
        print('[bwutils] output_type', output_type)
        print('[bwutils] cfa_pattern', cfa_pattern)
        print('[bwutils] patch_size', patch_size)
        print('[bwutils] crop_size', crop_size)
        print('[bwutils] upscaling_factor', upscaling_factor)
        print('[bwutils] input_max', input_max)
        print('[bwutils] loss_type', loss_type)
        print('[bwutils] loss_weight', loss_weight)
        print('[bwutils] loss_mode', loss_mode, self.loss_norm)
        print('[bwutils] loss_scale', loss_scale)
        print('[bwutils] cache_enable', cache_enable)




    def save_models(self, model, path, order):
        # init_variables()


        # to h5
        model.save(path + '_%s.h5' % order, include_optimizer=False)

        # model.input.set_shape(1 + model.input.shape[1:]) # to freeze model

        # to tflite
        # if tf.__version__.split('.')[0] == '1':
        #     print('tf version 1.x')
        #     converter = tf.lite.TFLiteConverter.from_keras_model_file(path + '_%s.h5' % order)
        #     tflite_model = converter.convert()
        #     open(path + '_%s.tflite' % order, "wb").write(tflite_model)
        # elif tf.__version__.split('.')[0] == '2':
        #     print('tf version 2.x')
        #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # tflite_model = converter.convert()
        # open(path + '_%s.tflite' % order, "wb").write(tflite_model)

        # # to yaml
        # model_string = model.to_yaml()
        # open(path + '.yaml', 'w').write(model_string)

        # to json
        model_json = model.to_json()
        open(path + '.json', 'w').write(model_json)



    def get_image_from_single_example(self, example, key='image', num_channels=3, dtype='uint8'):

        patch_size = self.patch_size

        feature = {
            key: tf.io.FixedLenFeature((), tf.string)
        }

        parsed = tf.io.parse_single_example(example, feature)

        image = tf.io.decode_raw(parsed[key], out_type=dtype)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [patch_size, patch_size, num_channels])

        return image


    def get_patternized_1ch_raw_image(self, image):
        patternized = self.get_patternized_3ch_image(image)
        patternized = tf.expand_dims(tf.reduce_sum(patternized, axis=-1), axis=-1)
        return patternized


    def get_patternized_3ch_image(self, image):
        tf_RGB = tf.constant(self.idx_RGB, dtype=tf.float32)
        patternized = tf.math.multiply(image[...,:3], tf_RGB)
        return patternized


    def get_patternized_4ch_image(self, image, is_shrink=True):

        if is_shrink:
            dtype = tf.bool
        else:
            dtype = tf.float32

        tf_G1 = tf.constant(self.idx_G1, dtype=dtype)
        tf_R  = tf.constant(self.idx_R,  dtype=dtype)
        tf_B  = tf.constant(self.idx_B,  dtype=dtype)
        tf_G2 = tf.constant(self.idx_G2, dtype=dtype)

        if is_shrink:
            if self.upscaling_factor == None:
                crop_size = self.crop_size
            else:
                crop_size = int( self.crop_size // self.upscaling_factor)
            G1 = tf.reshape(image[:, :, 1][tf_G1], (crop_size // 2, crop_size // 2))
            R  = tf.reshape(image[:, :, 0][tf_R],  (crop_size // 2, crop_size // 2))
            B  = tf.reshape(image[:, :, 2][tf_B],  (crop_size // 2, crop_size // 2))
            G2 = tf.reshape(image[:, :, 1][tf_G2], (crop_size // 2, crop_size // 2))
        else: # non_shrink
            G1 = tf.math.multiply(image[:, :, 1], tf_G1)
            R  = tf.math.multiply(image[:, :, 0], tf_R)
            B  = tf.math.multiply(image[:, :, 2], tf_B)
            G2 = tf.math.multiply(image[:, :, 1], tf_G2)


        pattenrized = tf.concat((tf.expand_dims(G1, axis=-1),
                                 tf.expand_dims(R, axis=-1),
                                 tf.expand_dims(B, axis=-1),
                                 tf.expand_dims(G2, axis=-1)),
                                axis=-1)

        return pattenrized

    def get_patternized_1ch_to_3ch_image(self, image):

        dtype = tf.float32

        tf_R = tf.constant(self.idx_R,  dtype=dtype)
        tf_G = tf.constant(self.idx_G, dtype=dtype)
        tf_B = tf.constant(self.idx_B,  dtype=dtype)

        # non_shrink
        R  = tf.math.multiply(image, tf_R)
        G  = tf.math.multiply(image, tf_G)
        B  = tf.math.multiply(image, tf_B)

        pattenrized = tf.concat((tf.expand_dims(R, axis=-1),
                                 tf.expand_dims(G, axis=-1),
                                 tf.expand_dims(B, axis=-1)),
                                axis=-1)

        return pattenrized



    def get_patternized(self, image, input_type='nonshrink'):

        print('hello patternized')
        # exit()

        if self.crop_size < self.patch_size:
            dim=3
            image = tf.image.random_crop(image, [self.crop_size, self.crop_size, dim])

        if input_type in ['shrink']:

            patternized = self.get_patternized_4ch_image(image, is_shrink=True)

        elif input_type in ['nonshrink_4ch']:

            patternized = self.get_patternized_4ch_image(image, is_shrink=False)

        elif input_type in ['raw_1ch']:

            patternized = self.get_patternized_1ch_raw_image(image)

        elif input_type in ['nonshrink']:

            patternized = self.get_patternized_3ch_image(image)

        else:
            ValueError('unknown input type', input_type)

        return patternized


    # scale_by_input_max
    def scale_by_input_max(self, image):
        image = image / self.input_max
        return image



    def add_noise_batch(self, inp, gt):
        '''
        image ~ (-1,1) normalized
        '''
        noise_gaussian_inp = tf.random.normal(tf.shape(inp)) / 255.
        noise_gaussian_gt = tf.random.normal(tf.shape(gt)) / 255.
        # noise_gaussian = self.scale_by_input_max(noise_gaussian)

        noise_poisson_inp = tf.random.normal(tf.shape(inp)) * tf.math.sqrt(tf.math.abs(inp)) / 255.
        noise_poisson_gt = tf.random.normal(tf.shape(gt)) * tf.math.sqrt(tf.math.abs(gt)) / 255.
        # noise_poisson = self.scale_by_input_max(noise_poisson)

        # hot bp in input image ~ 1%
        mask = tf.random.normal(tf.shape(inp)) > 2.33 # z=2.33 ~ .4901



        print('------------------------------------------')
        print('------------------------------------------')
        print('------------------------------------------')
        print('------------------------------------------')
        print('shape', noise_poisson_inp.shape, noise_poisson_gt.shape)

        inp = inp + noise_gaussian_inp + noise_poisson_inp
        inp = inp + tf.cast(mask, tf.float32)

        gt  = gt  + noise_gaussian_gt  + noise_poisson_gt

        inp = tf.clip_by_value(inp, -1, 1)
        gt  = tf.clip_by_value(gt , -1, 1)

        return inp, gt





    def data_augmentation(self, inp, gt):

        # flip
        if random.random() > 0.5:
            gt = tf.image.flip_left_right(gt)
            inp  = tf.image.flip_left_right(inp)

        if random.random() > 0.5:
            gt = tf.image.flip_up_down (gt)
            inp  = tf.image.flip_up_down (inp)

        # rotation
        r = np.random.randint(4)
        gt = tf.image.rot90(gt, r)
        inp  = tf.image.rot90(inp, r)

        return inp, gt

    def data_augmentation1(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, np.random.randint(4))
        return image

    def cure_static_bp(self, inp):
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)
        print("=============================, inp.shape", inp.shape)


        # inp = tf.make_ndarray(inp)
        # ## Red
        # for yy in range(1,inp.shape[0],4):
        #     for xx in range(1, inp.shape[1], 4):
        #         inp[yy][xx] = ((inp[yy-1][xx]+inp[yy][xx-1])/2)
        #
        # ## Blue
        # for yy in range(3,inp.shape[0],4):
        #     for xx in range(3, inp.shape[1], 4):
        #         inp[yy][xx] = ((inp[yy-1][xx]+inp[yy][xx-1])/2)

        # return tf.convert_to_tensor(inp)
        return inp

    def parse_tfrecord(self, example, mode):

        patch_size = self.patch_size

        ## get single image
        feature = {
            'gt':  tf.io.FixedLenFeature([], tf.string),
            'in': tf.io.FixedLenFeature([], tf.string)
        }

        parsed = tf.io.parse_single_example(example, feature)


        gt   = tf.io.decode_raw(parsed['gt'], 'uint8')
        inp  = tf.io.decode_raw(parsed['in'], 'uint16')

        # reshape
        gt   = tf.reshape(gt,  (patch_size, patch_size, 3))
        inp  = tf.reshape(inp, (patch_size, patch_size))

        # cast & normalize
        gt = tf.cast(gt, tf.float32) / (2**8 - 1)
        inp  = tf.cast(inp,  tf.float32) / (2**10 -1) ## <-- normalized to 1


        ## cure static bp
        # inp = self.cure_static_bp(inp)

        # raw 1ch to 3ch
        print('>>>>>>> inp.shape', inp.shape)
        inp = self.get_patternized_1ch_to_3ch_image(inp)
        print('<<<<<<< inp.shape', inp.shape)

        # augmentation
        if mode == tf.estimator.ModeKeys.TRAIN:
            gt, inp = self.data_augmentation(gt, inp)

        if self.input_bias:
             gt = (gt * 2) - 1
             inp  = (inp  * 2) - 1

        print('<<<<<<< gt.shape', gt.shape)

        return inp, gt

    def parse_tfrecord_single(self, example, mode):
        image = self.get_image_from_single_example(example, key='image', num_channels=3)

        if mode == tf.estimator.ModeKeys.TRAIN:
            image = self.data_augmentation1(image)

        image = self.scale_by_input_max(image) # normalized  (0, 1)
        image = tf.clip_by_value(image, 0, 1)

        if self.input_bias:
            image = (image * 2) - 1 # (0, 1) -->  (-1, 1)

        patternized = self.get_patternized(image, self.input_type)

        print('====================== patternized.shape ', patternized.shape)
        print('====================== image.shape ', image.shape)


        return patternized, image



    def dataset_input_fn(self, params):

        dataset = tf.data.TFRecordDataset(params['filenames'])
        if params['train_type'] == 'pair':
            parse_fn = self.parse_tfrecord
        elif params['train_type'] == 'single':
            parse_fn = self.parse_tfrecord_single
        else:
            ValueError("unknown mode, ", params['mode'])
            # exit()
        print('-------->, ', params['train_type'])
        dataset = dataset.map(partial(parse_fn, mode=params['mode']),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Dataset cache need 120G Main memory
        if self.cache_enable is True:
            dataset = dataset.cache()

        if params['mode'] == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(params['shuffle_buff']).repeat()

        dataset = dataset.batch(params['batch'])

        dataset = dataset.map(self.add_noise_batch)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


    def loss_fn(self, y_true, y_pred):

        loss = 0

        if ('rgb' in self.loss_type) and ('yuv' in self.loss_type):
            loss += self.loss_fn_mse_rgb_yuv(y_true, y_pred)
        elif 'rgb' in self.loss_type:
            loss += self.loss_fn_mse_rgb(y_true, y_pred)
        elif 'yuv' in self.loss_type:
            loss += self.loss_fn_mse_yuv(y_true, y_pred)
        if 'dct' in self.loss_type:
            loss += self.loss_fn_dct_2d(y_true, y_pred)
        if 'lattice' in self.loss_type:
            loss += self.loss_fn_fft_lattice(y_true, y_pred)
        if 'dir' in self.loss_type:
            loss += self.loss_fn_dir_follow(y_true, y_pred)


        if 'ploss' in self.loss_type:
            loss += self.loss_fn_ploss(y_true, y_pred)
        if 'ssim' in self.loss_type:
            loss += self.loss_fn_ssim(y_true, y_pred)

        # return loss * self.loss_scale

        return loss


    def loss_fn_mse_rgb(self, y_true, y_pred):
        print('rgb')
        rgb_mse_loss = tf.keras.backend.mean(self.loss_norm(y_true - y_pred))
        return rgb_mse_loss * self.loss_weight['rgb']


    def loss_fn_mse_yuv(self, y_true, y_pred):
        print('yuv')
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)

        yuv_mse_loss = tf.keras.backend.mean(
                tf.math.multiply(
                    tf.keras.backend.mean(self.loss_norm(y_true_yuv - y_pred_yuv), axis=[0, 1, 2]),
                    tf.constant([1., 2., 2], dtype=tf.float32)))

        return yuv_mse_loss * self.loss_weight['yuv']


    def loss_fn_mse_rgb_yuv(self, y_true, y_pred):
        print('rgb yuv')
        y_true_yuv = tf.image.rgb_to_yuv(y_true)
        y_pred_yuv = tf.image.rgb_to_yuv(y_pred)

        rgb_mse_loss = tf.keras.backend.mean(self.loss_norm(y_true - y_pred))
        yuv_mse_loss = tf.keras.backend.mean(
                tf.math.multiply(
                    tf.keras.backend.mean(self.loss_norm(y_true_yuv - y_pred_yuv), axis=[0, 1, 2]),
                    tf.constant([1., 2., 2], dtype=tf.float32)))

        return (rgb_mse_loss + yuv_mse_loss) * self.loss_weight['rgb']



    def loss_fn_ssim(self, y_true, y_pred):
        print('ssim')
        ssim_loss = 1. - tf.image.ssim(y_true, y_pred, 1)
        return ssim_loss


    def dct_2d(self, x):
        #x0 = tf.transpose(x, [0,3,1,2])  #BHWC-> BCHW
        x1 = tf.signal.dct(tf.transpose(x, [0,3,1,2]))
        x2 = tf.signal.dct(tf.transpose(x1, [0,1,3,2]))
        # xf = tf.transpose(x2, [0,3,2,1] )
        return x2
    def loss_fn_dct_2d(self, y_true, y_pred):
        print('dct2d')
        y_true_dct = self.dct_2d(y_true)
        y_pred_dct = self.dct_2d(y_pred)
        return tf.keras.losses.MeanAbsoluteError()(y_true_dct, y_pred_dct) * self.loss_weight['dct']


    def loss_fn_fft_lattice(self, y_true, y_pred):
        print('lattice')
        y_pred_fft_all=tf.signal.rfft2d(tf.transpose(y_pred, [0,3,1,2]))
        y_pred_fft_ext=tf.boolean_mask(y_pred_fft_all, self.tf_idx_fft)
        y_pred_fft_ext=tf.abs(y_pred_fft_ext)
        out=self.lattice_factor*tf.reduce_mean(tf.square(y_pred_fft_ext))
        return out * self.loss_weight['lattice']

    def basis_cal_dir(self, data, filter):
        return tf.nn.conv2d(data, filter, padding='VALID', strides=[1,1,1,1])
    def single_channel_cal_dir(self, data):
        H_vals = self.basis_cal_dir(data, self.tf_filter_H_dir)
        V_vals = self.basis_cal_dir(data, self.tf_filter_V_dir)
        S_vals = self.basis_cal_dir(data, self.tf_filter_S_dir)
        B_vals = self.basis_cal_dir(data, self.tf_filter_B_dir)
        P_vals = self.basis_cal_dir(data, self.tf_filter_P_dir)


        H_vals = tf.math.reduce_variance(H_vals[:, :, :, 0:5], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(H_vals[:, :, :, 5:10], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(H_vals[:, :, :, 10:15], axis=-1, keepdims=True)

        V_vals = tf.math.reduce_variance(V_vals[:, :, :, 0:5], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(V_vals[:, :, :, 5:10], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(V_vals[:, :, :, 10:15], axis=-1, keepdims=True)



        B_vals = tf.math.reduce_variance(B_vals[:, :, :, 0:5], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(B_vals[:, :, :, 5:9], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(B_vals[:, :, :, 9:13], axis=-1, keepdims=True)

        S_vals = tf.math.reduce_variance(S_vals[:, :, :, 0:5], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(S_vals[:, :, :, 5:9], axis=-1, keepdims=True) + \
                 tf.math.reduce_variance(S_vals[:, :, :, 9:13], axis=-1, keepdims=True)

        P_vals=tf.math.reduce_variance(P_vals)*3
        HVSBP_vals=tf.concat([P_vals, H_vals, V_vals, S_vals, B_vals], axis=-1)
        return HVSBP_vals
    def loss_fn_dir_follow(self, y_true, y_pred):
        print('dir')
        R_true_HVSBP=self.single_channel_cal_dir(y_true[:,:,:,0:1])
        G_true_HVSBP=self.single_channel_cal_dir(y_true[:,:,:,1:2])
        B_true_HVSBP=self.single_channel_cal_dir(y_true[:,:,:,2:3])

        R_pred_HVSBP=self.single_channel_cal_dir(y_pred[:,:,:,0:1])
        G_pred_HVSBP=self.single_channel_cal_dir(y_pred[:,:,:,1:2])
        B_pred_HVSBP=self.single_channel_cal_dir(y_pred[:,:,:,2:3])

        idx_R_true=tf.argmin(R_true_HVSBP, axis=-1)
        idx_R_true=tf.concat([tf.expand_dims(idx_R_true, axis=-1)==i for i in range(5)])
        idx_R_true=tf.cast(idx_R_true, dtype=tf.bool)

        R_true_vals = tf.boolean_mask(R_true_HVSBP, idx_R_true)
        R_pred_vals = tf.boolean_mask(R_pred_HVSBP, idx_R_true)

        G_true_vals = tf.boolean_mask(G_true_HVSBP, idx_R_true)
        G_pred_vals = tf.boolean_mask(G_pred_HVSBP, idx_R_true)

        B_true_vals = tf.boolean_mask(B_true_HVSBP, idx_R_true)
        B_pred_vals = tf.boolean_mask(B_pred_HVSBP, idx_R_true)


        out = 0
        out += tf.reduce_mean(tf.square(R_true_vals - R_pred_vals))
        out += tf.reduce_mean(tf.square(G_true_vals - G_pred_vals))
        out += tf.reduce_mean(tf.square(B_true_vals - B_pred_vals))

        return out * self.loss_weight['dir']



    def loss_fn_bayer(self, y_true, y_pred):

        y_true = self.get_patternized_1ch_raw_image(y_true)
        y_pred = self.get_patternized_1ch_raw_image(y_pred)

        bayer_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred))


        return bayer_loss

def get_checkpoint_manager(model, optimizer, directory, name):
    checkpoint = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                    directory=directory,
                                                    max_to_keep=5,
                                                    checkpoint_name=name,
                                                    )
    return checkpoint_manager


class BwCkptCallback(Callback):
    def __init__(self, model, optimizer, directory, name ):
        super().__init__()
        self.manager = get_checkpoint_manager(model, optimizer, directory, name)
        self.val_loss = np.inf


    def set_model(self, model):
        self.model = model

    def on_train_begin(self, _):
         pass

    def on_epoch_end(self, epoch, logs={}):
        print('--------------------> save gogo START')
        self.manager.save()
        print('--------------------> save gogo END')



class TensorBoardImage(Callback):
    def __init__(self, log_dir, dataloader, patch_size, cnt_viz, input_bias, cfa_pattern=1):
        super().__init__()
        self.log_dir = log_dir
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.cnt_viz = cnt_viz
        self.input_bias = input_bias



        self.idx_R = np.tile(
                np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                      np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G1 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G2 = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_B = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                                       np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

        self.idx_G = np.tile(
                np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                                np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
                  (patch_size // 2 // cfa_pattern, patch_size // 2 // cfa_pattern))

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(tag='Original Image', epoch=0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, tag, epoch):
        gidx = 0
        # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', len(list(self.dataloader)))
        # return
        for idx,  (x, y) in enumerate(self.dataloader):
            # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            # print('x.shape ', x.shape, ', y.shape ', y.shape)
            pred   = self.model(x)
            diff   = tf.math.abs(y-pred)

            if self.input_bias:
                x    = (   x + 1) / 2
                y    = (   y + 1) / 2
                pred = (pred + 1) / 2
                diff /= 2

            if x.shape[-1] == 1:
                tf_R  = tf.constant(self.idx_R,  dtype=np.float32)
                tf_G  = tf.constant(self.idx_G1, dtype=np.float32)
                tf_B  = tf.constant(self.idx_B,  dtype=np.float32)


                R = tf.math.multiply(x[...,0], tf_R)
                G = tf.math.multiply(x[...,0], tf_G)
                B = tf.math.multiply(x[...,0], tf_B)


                x = tf.concat((tf.expand_dims(R, axis=-1),
                               tf.expand_dims(G, axis=-1),
                               tf.expand_dims(B, axis=-1)),
                                axis=-1)


            print('x.shape', x.shape)
            print('y.shape', y.shape)
            print('y.shape', y.shape)
            print('y.shape', y.shape)

            all_images = tf.concat( [tf.concat([x, y]      , axis=2),
                                     tf.concat([diff, pred], axis=2)] , axis=1)

            print('x: %.2f, %.2f' %( tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy()), end='')
            print(', y: %.2f, %.2f' %( tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy()), end='')
            print(', pred: %.2f, %.2f' % (  tf.reduce_min(pred).numpy(), tf.reduce_max(pred).numpy()), end='')
            print(', diff: %.2f, %.2f' % ( tf.reduce_min(diff).numpy(), tf.reduce_max(diff).numpy()))

            with self.writer.as_default():
                tf.summary.image(f"Viz set {gidx}", all_images, max_outputs=16, step=epoch)
            gidx+=1

        self.writer.flush()

    def on_epoch_end(self, epoch, logs={}):
        self.write_image('Images', epoch)

def get_training_callbacks(names, base_path, model_name=None, dataloader=None, patch_size=128, cnt_viz=4, input_bias=True, initial_value_threshold=np.inf):
    callbacks=[]
    # callbacks =  tf.keras.callbacks.CallbackList()
    if 'ckeckpoint' in names:
        ckpt_dir = os.path.join(base_path, 'checkpoint', model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckeckpoint_dir = os.path.join(ckpt_dir, '{epoch:05d}_%s_{loss:.5e}.h5' % (model_name))
        callback_ckpt = tf.keras.callbacks.ModelCheckpoint(
                                filepath = ckeckpoint_dir,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=False,
                                save_weights_only=False,
                                initial_value_threshold=initial_value_threshold )
        callbacks.append(callback_ckpt)
    if 'tensorboard' in names:
        tb_dir = os.path.join(base_path, 'board', model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_tb = tf.keras.callbacks.TensorBoard( log_dir=tb_dir,
                                                       histogram_freq=10,
                                                       write_graph=True,
                                                       write_images=False)
        callbacks.append(callback_tb)
    if 'image' in names:
        tb_dir = os.path.join(base_path, 'board', model_name, 'image') #, model_name)
        os.makedirs(tb_dir, exist_ok=True)
        callback_images =TensorBoardImage( log_dir=tb_dir,
                                           dataloader = dataloader,
                                           patch_size = patch_size,
                                           cnt_viz = cnt_viz,
                                           input_bias=input_bias)
        callbacks.append(callback_images)
    return callbacks



def get_scheduler(type='cosine', lr_init=2e-3, lr_last=1e-5, steps=100):
    print('-------------------> scheduler type, ', type)

    if float(tf.__version__[:-2]) > 2.4:
        if type.lower() == 'cosine':
            print('-------------------> GOOD scheduler type1, ', type)
            scheduler =tf.keras.optimizers.schedules.CosineDecay(
                                initial_learning_rate=lr_init,
                                decay_steps=steps,
                                alpha=0.5,
                                name='CosineDecay')
        else:
            print('-------------------> WTF scheduler type1, ', type, type =='cosine')
            scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
                                initial_learning_rate=lr_init,
                                first_decay_steps=steps,
                                t_mul=2.0,
                                m_mul=1.0,
                                alpha=lr_last,
                                name='CosineDecayRestarts')
    else:
        if type.lower() == 'cosine':
            print('-------------------> GOOD scheduler type2, ', type)
            scheduler =tf.keras.experimental.CosineDecay(
                                initial_learning_rate=lr_init,
                                decay_steps=steps,
                                alpha=0.5,
                                name='CosineDecay')
        else:
            print('-------------------> WTF scheduler type2, ', type)
            scheduler = tf.keras.experimental.CosineDecayRestarts(
                                initial_learning_rate=lr_init,
                                first_decay_steps=steps,
                                t_mul=2.0,
                                m_mul=1.0,
                                alpha=lr_last,
                                name='CosineDecayRestarts')


    lr_callback =  tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    return lr_callback




if __name__ == '__main__':

    butils = bwutils('shrink', 'tetra')



    print('done')
