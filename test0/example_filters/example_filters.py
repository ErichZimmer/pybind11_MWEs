import example_filters_bindings

def gaussian_filter(img, kernel_size=3, sigma=1):
    # store original image dtype
    img_dtype = img.dtype
    
    max_ = img.max()
    if max_ > 1: 
        img = img.astype("float32")
        img /= max_
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = np.pad(img, pad)
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # extract filtered image
    new_img = example_filters_bindings.gaussian_filter(buffer1, kernel_size, sigma)#[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
        new_img = new_img.astype(img_dtype)
      
    return new_img
    
def local_gaussian_variance_filter(img, kernel_size=3, sigma1=2, sigma2=2):
    # store original image dtype
    img_dtype = img.dtype
    
    max_ = img.max()
    if max_ > 1: 
        img = img.astype("float32")
        img /= max_
    
    # pad array by kernel half size
    pad = int(kernel_size / 2)
    buffer1 = np.pad(img, pad)
    
    # make sure array is float32
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    
    # extract filtered image
    new_img = example_filters_bindings.local_variance_normalization(buffer1, kernel_size, sigma)[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
    
    # if the image wasn't normalized beforehand, return original range
    if max_ > 1: 
        new_img *= max_
        new_img = new_img.astype(img_dtype)
      
    return new_img