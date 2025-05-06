import numpy as np
import matplotlib.pyplot as plt

cfa = 2
pad = cfa*(4-1)

tetra = np.arange(64).reshape(8,8)
# tetra = np.array([['R', 'R', 'G', 'G', 'R', 'R', 'G', 'G'],
#                   ['R', 'R', 'G', 'G', 'R', 'R', 'G', 'G'],
#                   ['G', 'G', 'B', 'B', 'G', 'G', 'B', 'B'],
#                   ['G', 'G', 'B', 'B', 'G', 'G', 'B', 'B'],
#                   ['R', 'R', 'G', 'G', 'R', 'R', 'G', 'G'],
#                   ['R', 'R', 'G', 'G', 'R', 'R', 'G', 'G'],
#                   ['G', 'G', 'B', 'B', 'G', 'G', 'B', 'B'],
#                   ['G', 'G', 'B', 'B', 'G', 'G', 'B', 'B'],
#                   ])
# print(tetra)



# tetra_pad_w = np.pad(tetra, 6, 'symmetric')


# tetra_pad_w = np.concatenate([tetra[pad-1:cfa-1:-1],
#                               tetra,
#                               tetra[-cfa-1:-pad-1:-1]], axis=0)
#
# tetra_pad_h = np.concatenate([tetra[:,pad-1:cfa-1:-1],
#                               tetra,
#                               tetra[:,-cfa-1:-pad-1:-1]], axis=1)



print(tetra_pad_w, tetra_pad_w.shape)
# print(tetra_pad_h, tetra_pad_h.shape)




