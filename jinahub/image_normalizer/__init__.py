__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Union, Iterable
import numpy as np

from jina import DocumentArray, Executor, requests
from .helper import _load_image, _move_channel_axis, _crop_image, _resize_short


class ImageNormalizer(Executor):
    def __init__(
        self,
        target_size: Union[Iterable[int], int] = 224,
        img_mean: Tuple[float] = (0, 0, 0),
        img_std: Tuple[float] = (1, 1, 1),
        resize_dim: Union[Iterable[int], int] = 256,
        channel_axis: int = -1,
        target_channel_axis: int = -1,
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

    def craft(self, docs: DocumentArray, fn) -> DocumentArray:
        filtered_docs = DocumentArray(
            list(
                filter(lambda d: 'image/' in d.mime_type, docs)
            )
        )
        for doc in filtered_docs:
            getattr(doc, fn)()
            raw_img = _load_image(doc.blob, self.channel_axis)
            _img = self._normalize(raw_img)
            # move the channel_axis to target_channel_axis to better fit
            # different models
            img = _move_channel_axis(_img, -1, self.target_channel_axis)
            doc.blob = img
        return filtered_docs

    @requests(on='/index')
    def craft_index(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        return self.craft(docs, 'convert_image_uri_to_blob')

    @requests(on='/search')
    def craft_search(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        return self.craft(docs, 'convert_image_datauri_to_blob')

    def _normalize(self, img):
        img = _resize_short(img, target_size=self.resize_dim)
        img, _, _ = _crop_image(img, target_size=self.target_size, how='center')
        img = np.array(img).astype('float32') / 255
        img -= self.img_mean
        img /= self.img_std
        return img