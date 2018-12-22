
from .common_utils import *

def bayer_mosaic(img_np, pattern='rggb'):
    
    c, w, h = img_np.shape

    # Create zero array, same as size of the original image
    mask_np = np.zeros((1, w, h), dtype=np.float32)

    # Map the RGB values in the original picture according to the pattern
    if pattern=='rggb':
        # Red
        mask_np[0, 0::2, 0::2] = img_np[0, 0::2, 0::2]
        # Green (top row of the Bayer matrix)
        mask_np[0, 0::2, 1::2] = img_np[1, 0::2, 1::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[0, 1::2, 0::2] = img_np[1, 1::2, 0::2]
        # Blue
        mask_np[0, 1::2, 1::2] = img_np[2, 1::2, 1::2]
    elif pattern=='grbg':
        # Green (top row of the Bayer matrix)
        mask_np[0, 0::2, 0::2] = img_np[1, 0::2, 0::2]
        # Red
        mask_np[0, 0::2, 1::2] = img_np[0, 0::2, 1::2]
        # Blue
        mask_np[0, 1::2, 0::2] = img_np[2, 1::2, 0::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[0, 1::2, 1::2] = img_np[1, 1::2, 1::2]
    elif pattern=='bggr':
        # Blue
        mask_np[0, 0::2, 0::2] = img_np[2, 0::2, 0::2]
        # Green (top row of the Bayer matrix)
        mask_np[0, 0::2, 1::2] = img_np[1, 0::2, 1::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[0, 1::2, 0::2] = img_np[1, 1::2, 0::2]
        # Red
        mask_np[0, 1::2, 1::2] = img_np[0, 1::2, 1::2]
    elif pattern=='gbrg':
        # Green (top row of the Bayer matrix)
        mask_np[0, 0::2, 0::2] = img_np[1, 0::2, 0::2]
        # Blue
        mask_np[0, 0::2, 1::2] = img_np[2, 0::2, 1::2]
        # Red
        mask_np[0, 1::2, 0::2] = img_np[0, 1::2, 0::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[0, 1::2, 1::2] = img_np[1, 1::2, 1::2]
    else:
        assert False, 'Invalid pattern'
            
    return mask_np

def bayer_mosaic_torch(img_np,dtype, pattern='rggb'):
    
    a, c, w, h = img_np.shape

    # Create zero tensor, same as size of the original image
    mask_np = torch.zeros((a, 1, w, h)).type(dtype)

    # Map the RGB values in the original picture according to the pattern
    if pattern=='rggb':
        # Red
        mask_np[:,0, 0::2, 0::2] = img_np[:,0, 0::2, 0::2]
        # Green (top row of the Bayer matrix)
        mask_np[:,0, 0::2, 1::2] = img_np[:,1, 0::2, 1::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[:,0, 1::2, 0::2] = img_np[:,1, 1::2, 0::2]
        # Blue
        mask_np[:,0, 1::2, 1::2] = img_np[:,2, 1::2, 1::2]
    elif pattern=='grbg':
        # Green (top row of the Bayer matrix)
        mask_np[:,0, 0::2, 0::2] = img_np[:,1, 0::2, 0::2]
        # Red
        mask_np[:,0, 0::2, 1::2] = img_np[:,0, 0::2, 1::2]
        # Blue
        mask_np[:,0, 1::2, 0::2] = img_np[:,2, 1::2, 0::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[:,0, 1::2, 1::2] = img_np[:,1, 1::2, 1::2]
    elif pattern=='bggr':
        # Blue
        mask_np[:,0, 0::2, 0::2] = img_np[:,2, 0::2, 0::2]
        # Green (top row of the Bayer matrix)
        mask_np[:,0, 0::2, 1::2] = img_np[:,1, 0::2, 1::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[:,0, 1::2, 0::2] = img_np[:,1, 1::2, 0::2]
        # Red
        mask_np[:,0, 1::2, 1::2] = img_np[:,0, 1::2, 1::2]
    elif pattern=='gbrg':
        # Green (top row of the Bayer matrix)
        mask_np[:,0, 0::2, 0::2] = img_np[:,1, 0::2, 0::2]
        # Blue
        mask_np[:,0, 0::2, 1::2] = img_np[:,2, 0::2, 1::2]
        # Red
        mask_np[:,0, 1::2, 0::2] = img_np[:,0, 1::2, 0::2]
        # Green (bottom row of the Bayer matrix)
        mask_np[:,0, 1::2, 1::2] = img_np[:,1, 1::2, 1::2]
    else:
        assert False, 'Invalid pattern'
            
    return mask_np


def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))
