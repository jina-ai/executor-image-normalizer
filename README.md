<p align="center">
<img src="https://github.com/jina-ai/jina/blob/master/.github/logo-only.gif?raw=true" alt="Jina banner" width="200px">
</p>

# Image Normalizer

### Description
The image normalizer resizes, crops and normalizes images.

### Parameters
The following parameters can be used:

- `resize_dim` (int): The size of the image after resizing
- `target_size` (tuple or int): The dimensions to crop the image to (center cropping is used)
- `img_mean` (tuple, default (0,0,0)): The mean for normalization
- `img_std` (tuple, default (1,1,1)): The standard deviation for normalization
- `channel_axis` (int): The channel axis in the images used
- `target_channel_axis` (int): The desired channel axis in the images. If this is not equal to the channel_axis, the axis is moved.

