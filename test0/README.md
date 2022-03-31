## Convolution kernels test
Example if an implemetation of convolution kernels for image filters. The filters assume the image will be padded by half the width of the kernel size and normalized to [0, 1] 32 bit float.
An example implementation of the filter would look like:
```python
import numpy as np
import example_filters
def gaussian_filter(img, kernel, sigma):
    pad = int(kernel / 2)
    buffer1 = np.pad(img, ((pad, pad), (pad, pad)))
    if buffer1.dtype != "float32":
        buffer1 = buffer1.astype("float32")
    out = np.zeros_like(buffer1, dtype = "float32")
    example_filters.gaussian_filter(out, buffer1, kernel, sigma)
    return out[pad : buffer1.shape[0]-pad, pad : buffer1.shape[1]-pad]
```

##To-Do:
[ ] Breakdown main to several files
[ ] Add more convolution filters
[ ] Make header files work
