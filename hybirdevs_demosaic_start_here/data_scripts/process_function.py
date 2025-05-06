import numpy as np
import os


def read_bin(img_path):
    """
    Read bin data
    """
    img_data = np.fromfile(img_path, dtype=np.uint16)
    w = int(img_data[0])
    h = int(img_data[1])
    assert w * h == img_data.size - 2
    quad = np.clip(img_data[2:].reshape([h, w]).astype(np.float32), 0, 1023)
    return quad


if __name__ == '__main__':
    img_path = r'./sample/0001.bin'
    input_quad = read_bin(img_path)



