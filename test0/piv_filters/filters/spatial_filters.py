"""
Module for filtering Particle Image Velocimetry (PIV) images to attain a higher quality image.
"""
from piv_filters_core import _intensity_cap, _threshold_binarization,\
    _gaussian_lowpass_filter, _gaussian_highpass_filter, _local_variance_normalization,\
    _test_wrapper
from numpy import pad as pad_array, percentile

kernel_size_error = "kernel_size must be an odd number"

def instensity_cap(img, std_mult = 2.0):
    """
    Set pixels higher than calculated threshold to said threshold. 
    The threshold value is calcualted by img_mean + std_mult * img_std.
    
    Parameters
    ----------
    img: 2d np.ndarray
        a two dimensional array containing pixel intenensities.
        
    std_mult: float
        Lower values yields a lower threshold
        
    Returns
    -------
    img: 2d np.ndarray
        a filtered two dimensional array of the input image
    """
    # store original image dtype
    img_dtype = img.dtype
    
    if img_dtype != "float32":
        img = img.astype("float32")
        
    new_img = _intensity_cap(
        img,
        float(std_mult)
    )
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    return new_img


def threshold_binarization(img, threshold=0.5):
    """
    A threshold binarization filter.
    
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
    
    # make sure array is float32
    if img.dtype != "float32":
        img = img.astype("float32")
    
    # normalize array if pixel intensities are greater than 1
    if max_ > 1: 
        img = img.astype("float32")
        img /= max_
        
    # extract filtered image
    new_img = _threshold_binarization(
        img,
        float(threshold)
    )
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    return new_img


def gaussian_filter(img, kernel_size=3, sigma=1.0):
    """
    A simple sliding window gaussian low pass filter.
    
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
    
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
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
    new_img = _gaussian_lowpass_filter(
        buffer1,
        kernel_size, 
        float(sigma)
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


def highpass_filter(img, kernel_size=3, sigma=1.0, clip_at_zero=True):
    """
    A simple sliding window gaussian high pass filter.
    
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
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
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
    new_img = _gaussian_highpass_filter(
        buffer1,
        kernel_size, 
        float(sigma),
        bool(clip_at_zero)
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_

    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


def variance_normalization_filter(img, kernel_size=3, sigma1=1.0, sigma2=1.0, clip_at_zero=True):
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
    if kernel_size % 2 != 1:
        raise Exception(kernel_size_error)
        
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
    new_img = _local_variance_normalization(
        buffer1, 
        kernel_size,
        float(sigma1), float(sigma2),
        bool(clip_at_zero)
    )
    
    # remove padding
    new_img = new_img[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
    
    if img_dtype != "float32":
        new_img = new_img.astype(img_dtype)
        
    if flip == True:
        new_img = new_img.T
        
    return new_img


try
    from skimage.exposure import rescale_intensity
    def contrast_stretch(img, lower_limit = 2, upper_limit = 98):
        """
        Simple percentile-based contrast stretching.

        Parameters
        ----------

        img : 2d np.ndarray
            a two dimensional array containing pixel intenensities.

        lower_limit: int
            lower percentile limit

        upper_limit: int
            upper percentile limit

        Returns
        -------
        img: image
            a filtered two dimensional array of the input image  
        """
        if lower_limit < 0:
            lower_limit = 0
        if upper_limit > 100:
            upper_limit = 100

        lower = percentile(img, lower_limit)
        upper = percentile(img, upper_limit)
        img = rescale_intensity(img, in_range = (lower, upper))
        return img
    
    print("Loaded rescale_intensity function")
except ImportError:
    pass