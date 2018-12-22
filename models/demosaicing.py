import numpy as np
import torch
import torch.nn as nn 
# from .common_utils import *

class Mosaic(nn.Module):
    
    def __init__(self, 
    
    
    
    
    
    
def bayer_mosaic(img_np, pattern='rggb'):
    
#     w, h, c = img_np.shape

    # Create target array, same as size of the original image
    mask_np = np.zeros((img_np.shape), dtype=np.uint8)

    # Map the RGB values in the original picture according to the BGGR pattern# 
    if pattern=='rggb':
        # Blue
        mask_np[1::2, 1::2,2] = img_np[1::2, 1::2,0]
        # Green (top row of the Bayer matrix)
        mask_np[ 0::2, 1::2,1] = img_np[0::2, 1::2,0]
        # Green (bottom row of the Bayer matrix)
        mask_np[ 1::2, 0::2,1] = img_np[1::2, 0::2,0]
        # Red
        mask_np[ 0::2, 0::2,0] = img_np[0::2, 0::2,0]
    else:
        assert False, 'Invalid pattern'
            
    return mask_np
