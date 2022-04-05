#from pybind11_test2.example_prepoc_filters import gaussian_filter, local_variance_normalization
from example_filters_bindings import gaussian_filter, local_variance_normalization
from numpy import pad as pad_array

def gaussian_filter(img, kernel_size=3, sigma=1):
    """
    A simple sliding window gaussian filter.
    
    Parameters
    ----------
    
    img : 2d np.ndarray
        a two dimensional array containing pixel intenensities.
        
    kernel_size : int
        nxn size of the convolution kernel
    
    sigma : float
        sigma of nxn gaussian convolution kernel
        
    Returns
    -------
    
    new_img : 2d np.ndarray
        a two dimensional array containing pixel intenensities.
    """
    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()
    flip = False
    
    # check if image needs to be flipped
    if img.shape[0] > img.shape[1]:
        flip = True
        img = img.T
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = pad_array(img, pad, mode = "reflect")
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_
        
    # extract filtered image
    new_img = gaussian_filter(
        buffer1,
        kernel_size, 
        sigma
    )
    
#    new_img = buffer1

    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
        new_img = new_img.astype(img_dtype)
    
    if flip == True:
        new_img = new_img.T
        
    return new_img


def local_gaussian_variance_filter(img, kernel_size=3, sigma=1):
    """
    A simple gaussian variance normalization filter.
    
    Parameters
    ----------
    
    img : 2d np.ndarray
        a two dimensional array containing pixel intenensities.
        
    kernel_size : int
        nxn size of the convolution kernel
    
    sigma1 : float
        sigma of nxn gaussian convolution kernel
    
    sigma2 : float
        sigma of nxn gaussian convolution kernel
        
    Returns
    -------
    
    new_img : 2d np.ndarray
        a two dimensional array containing pixel intenensities.
    """
    # store original image dtype
    img_dtype = img.dtype
    max_ = img.max()
    flip = False
    
    # check if image needs to be flipped
    if img.shape[0] > img.shape[1]:
        flip = True
        img = img.T
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = pad_array(img, pad, mode = "reflect")
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        buffer1 = buffer1.astype("float32")
        buffer1 /= max_
        
    # extract filtered image
    new_img = local_variance_normalization(
        buffer1, 
        kernel_size,
        sigma1, sigma2
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
        new_img = new_img.astype(img_dtype)
    
    if flip == True:
        new_img = new_img.T
        
    return new_img