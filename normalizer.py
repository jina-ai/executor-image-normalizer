__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import torch
from jina import DocumentArray, Executor, requests
from torchvision import transforms

Tensor = torch.Tensor


class ImageNormalizer(Executor):
    def __init__(
        self,
        target_size: Union[Iterable[int], int] = 224,
        img_mean: Tuple[float] = (0, 0, 0),
        img_std: Tuple[float] = (1, 1, 1),
        resize_dim: Union[Iterable[int], int] = 256,
        channel_axis: int = -1,
        target_channel_axis: int = -1,
        image_smoothing: str = None,
        local_response_norm: bool = False,
        local_contrast_norm: bool = False,
        *args,
        **kwargs,
    ):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.resize_dim = resize_dim
        self.img_mean = np.array(img_mean).reshape((1, 1, 3))
        self.img_std = np.array(img_std).reshape((1, 1, 3))
        self.channel_axis = channel_axis
        self.target_channel_axis = target_channel_axis
        self.image_smoothing = image_smoothing
        if self.image_smoothing:
            self.error_msg = (
                f"Image smoothing is either 'gaussian', 'averaging'"
                f"'median' or 'bilateral', got {self.image_smoothing}."
            )
            assert self.image_smoothing in [
                "gaussian",
                "averaging",
                "median",
                "bilateral",
            ], self.error_msg

        self.local_response_norm = local_response_norm
        self.local_contrast_norm = local_contrast_norm

    @requests
    def craft(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        filtered_docs = DocumentArray(
            list(filter(lambda d: 'image/' in d.mime_type, docs))
        )
        for doc in filtered_docs:
            raw_img = self._load_image(doc.blob)
            _img = self._normalize(raw_img)
            # move the channel_axis to target_channel_axis to better fit
            # different models
            img = self._move_channel_axis(_img, -1, self.target_channel_axis)
            doc.blob = img
        return filtered_docs

    def _normalize(self, img):
        """
        Apply resize, crop, smooth, local contrast and local response normalization

        Args:
        - img: image to normalize

        """
        img = self._resize_short(img)
        img, _, _ = self._crop_image(img, how='center')
        img = (
            self._image_smoothing(img, self.image_smoothing)
            if self.image_smoothing
            else img
        )
        if self.local_contrast_norm or self.local_response_norm:
            img_tensor = transforms.ToTensor()(img).unsqueeze_(0)
            img_tensor = (
                self._local_contrast_norm(img_tensor)
                if self.local_contrast_norm
                else img_tensor
            )
            img_tensor = (
                self._local_response_norm(img_tensor)
                if self.local_response_norm
                else img_tensor
            )
            img = transforms.ToPILImage()(img_tensor.squeeze_(0))

        img = np.array(img).astype('float32') / 255
        img -= self.img_mean
        img /= self.img_std
        return img

    def _image_smoothing(
        self,
        img: Image,
        image_smoothing: str = None,
        ksize: Tuple = (65, 65),
        sigmaXGaussian: int = 10,
        ksizeMedianBlur: int = 5,
        diameterBilateralFilter: int = 9,
        sigmaColorBilateralFilter: int = 75,
        sigmaSpaceBilateralFilter: int = 75,
    ) -> Image:
        """
        Apply image smoothing techniques to help reduce the noise by removing high frequency
        content (eg: noise, edges) from the image.
        We use OpenCV to provide four main types of smoothing techniques:
        - Averaging
        - Gaussian Blurring
        - Median Blurring
        - Bilateral Filtering

        Args:
            img: input image to be smoothed
            image_smoothing: image soothing technique
            ksize: Gaussian kernel size, ksize.width and ksize.height
            sigmaXGaussian: Gaussian kernel standard deviation in X direction.
            ksizeMedianBlur: Median blurring kernel size
            diameterBilateralFilter: Diameter of each pixel neighborhood that is used during filtering.
            sigmaColorBilateralFilter: Filter sigma in the color space.
            sigmaSpaceBilateralFilter: Filter sigma in the coordinate space.
        """
        img = np.array(img)
        if image_smoothing == "gaussian":
            image_smooth = cv2.GaussianBlur(img, ksize, sigmaXGaussian)
        elif image_smoothing == "averaging":
            image_smooth = cv2.blur(img, ksize)
        elif image_smoothing == "median":
            image_smooth = cv2.medianBlur(img, ksizeMedianBlur)
        elif image_smoothing == "bilateral":
            image_smooth = cv2.bilateralFilter(
                img,
                diameterBilateralFilter,
                sigmaColorBilateralFilter,
                sigmaSpace_bilateralFilter,
            )
        else:
            assert self.error_msg

        image_smooth = cv2.subtract(img, image_smooth, dtype=cv2.CV_32F)
        return image_smooth

    def _local_response_norm(
        self,
        input: Tensor,
        size: int = 3,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ) -> Tensor:
        """
        Apply local response normalization over an input signal composed of
        several input planes, where channels occupy the second dimension.

        See :class:`~torch.nn.LocalResponseNorm` for details.
        """

        input = torch.nn.functional.local_response_norm(
            input, size, alpha=alpha, beta=beta, k=k
        )

        return input

    def _local_contrast_norm(self, input: Tensor, radius: int = 9) -> Tensor:
        """
        Apply local contrast normalization over an input image

        Args:
        - input: torch.Tensor , .shape => (1,channels,height,width)
        - radius: Gaussian filter size (int), odd
        """
        if radius % 2 == 0:
            radius += 1

        def get_gaussian_filter(kernel_shape):
            x = np.zeros(kernel_shape, dtype='float64')

            def gauss(x, y, sigma=2.0):
                Z = 2 * np.pi * sigma ** 2
                return 1.0 / Z * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

            mid = np.floor(kernel_shape[-1] / 2.0)
            for kernel_idx in range(0, kernel_shape[1]):
                for i in range(0, kernel_shape[2]):
                    for j in range(0, kernel_shape[3]):
                        x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

            return x / np.sum(x)

        n, c, h, w = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        gaussian_filter = Tensor(get_gaussian_filter((1, c, radius, radius)))
        filtered_out = torch.nn.functional.conv2d(
            input, gaussian_filter, padding=radius - 1
        )
        mid = int(np.floor(gaussian_filter.shape[2] / 2.0))
        ### Subtractive Normalization
        centered_image = input - filtered_out[:, :, mid:-mid, mid:-mid]

        ## Variance Calc
        sum_sqr_image = torch.nn.functional.conv2d(
            centered_image.pow(2), gaussian_filter, padding=radius - 1
        )
        s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
        per_img_mean = s_deviation.mean()

        ## Divisive Normalization
        divisor = np.maximum(per_img_mean.numpy(), s_deviation.numpy())
        divisor = np.maximum(divisor, 1e-4)
        new_image = centered_image / Tensor(divisor)
        return new_image

    def _load_image(self, blob: 'np.ndarray') -> Image:
        """
        Load an image array and return a `PIL.Image` object.
        """
        img = self._move_channel_axis(blob, self.channel_axis)
        return Image.fromarray(img.astype('uint8'))

    @staticmethod
    def _move_channel_axis(
        img: 'np.ndarray', channel_axis_to_move: int, target_channel_axis: int = -1
    ) -> 'np.ndarray':
        """
        Ensure the color channel axis is the default axis.
        """
        if channel_axis_to_move == target_channel_axis:
            return img
        return np.moveaxis(img, channel_axis_to_move, target_channel_axis)

    def _crop_image(self, img, top: int = None, left: int = None, how: str = 'precise'):
        """
        Crop the input :py:mod:`PIL` image.
        :param img: :py:mod:`PIL.Image`, the image to be resized
        :param target_size: desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the output will have the same height and width as the `target_size`.
        :param top: the vertical coordinate of the top left corner of the crop box.
        :param left: the horizontal coordinate of the top left corner of the crop box.
        :param how: the way of cropping. Valid values include `center`, `random`, and, `precise`. Default is `precise`.
            - `center`: crop the center part of the image
            - `random`: crop a random part of the image
            - `precise`: crop the part of the image specified by the crop box with the given ``top`` and ``left``.
            .. warning:: When `precise` is used, ``top`` and ``left`` must be fed valid value.
        """
        assert isinstance(img, Image.Image), 'img must be a PIL.Image'
        img_w, img_h = img.size
        if isinstance(self.target_size, int):
            target_h = target_w = self.target_size
        elif isinstance(self.target_size, Tuple) and len(self.target_size) == 2:
            target_h, target_w = self.target_size
        else:
            raise ValueError(
                f'target_size should be an integer or a tuple of '
                f'two integers: {self.target_size}'
            )
        w_beg = left
        h_beg = top
        if how == 'center':
            w_beg = int((img_w - target_w) / 2)
            h_beg = int((img_h - target_h) / 2)
        elif how == 'random':
            w_beg = np.random.randint(0, img_w - target_w + 1)
            h_beg = np.random.randint(0, img_h - target_h + 1)
        elif how == 'precise':
            assert w_beg is not None and h_beg is not None
            assert (
                0 <= w_beg <= (img_w - target_w)
            ), f'left must be within [0, {img_w - target_w}]: {w_beg}'
            assert (
                0 <= h_beg <= (img_h - target_h)
            ), f'top must be within [0, {img_h - target_h}]: {h_beg}'
        else:
            raise ValueError(f'unknown input how: {how}')
        if not isinstance(w_beg, int):
            raise ValueError(f'left must be int number between 0 and {img_w}: {left}')
        if not isinstance(h_beg, int):
            raise ValueError(f'top must be int number between 0 and {img_h}: {top}')
        w_end = w_beg + target_w
        h_end = h_beg + target_h
        img = img.crop((w_beg, h_beg, w_end, h_end))
        return img, h_beg, w_beg

    def _resize_short(self, img, how: str = 'LANCZOS'):
        """
        Resize the input :py:mod:`PIL` image.
        :param img: :py:mod:`PIL.Image`, the image to be resized
        :param target_size: desired output size. If size is a sequence like (h, w), the output size will be matched to
            this. If size is an int, the smaller edge of the image will be matched to this number maintain the aspect
            ratio.
        :param how: the interpolation method. Valid values include `NEAREST`, `BILINEAR`, `BICUBIC`, and `LANCZOS`.
            Default is `LANCZOS`. Please refer to `PIL.Image` for detaisl.
        """
        assert isinstance(img, Image.Image), 'img must be a PIL.Image'
        if isinstance(self.resize_dim, int):
            percent = float(self.resize_dim) / min(img.size[0], img.size[1])
            target_w = int(round(img.size[0] * percent))
            target_h = int(round(img.size[1] * percent))
        elif isinstance(self.resize_dim, Tuple) and len(self.resize_dim) == 2:
            target_w, target_h = self.resize_dim
        else:
            raise ValueError(
                f'target_size should be an integer or a tuple of two '
                f'integers: {self.resize_dim}'
            )
        img = img.resize((target_w, target_h), getattr(Image, how))
        return img
