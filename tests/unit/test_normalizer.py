import pytest
import os
import numpy as np
from PIL.Image import Image

from jina import DocumentArray, Document

from jinahub.image.normalizer import ImageNormalizer

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_image_uri_doc():
    return Document(uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))


@pytest.fixture
def test_image_buffer_doc():
    doc = Document(uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))
    doc.convert_uri_to_buffer()
    return doc


@pytest.fixture
def test_image_blob_doc():
    doc = Document(uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))
    doc.convert_image_uri_to_blob()
    return doc


def test_initialization():
    norm = ImageNormalizer()
    assert norm.target_size == 224
    norm = ImageNormalizer(
        target_size=96,
        img_mean=(1., 2., 3.),
        img_std=(2., 2., 2.),
        resize_dim=256,
        channel_axis=4,
        target_channel_axis=5,
    )
    assert norm.target_size == 96
    assert np.array_equal(norm.img_std, [[[2, 2, 2]]])
    assert np.array_equal(norm.img_mean, [[[1, 2, 3]]])
    assert norm.resize_dim == 256
    assert norm.channel_axis == 4
    assert norm.target_channel_axis == 5


def test_reading_image(test_image_uri_doc,
                       test_image_buffer_doc,
                       test_image_blob_doc):

    norm = ImageNormalizer(resize_dim=123,
                           img_mean=(0.1, 0.1, 0.1),
                           img_std=(0.5, 0.5, 0.5))

    docs = DocumentArray([test_image_uri_doc,
                          test_image_buffer_doc,
                          test_image_blob_doc])

    assert docs[0].blob is None and docs[1].blob is None
    norm.read(docs)
    assert len(docs) == 3
    for doc in docs:
        assert np.array_equal(doc.blob, test_image_blob_doc.blob)

    
@pytest.mark.parametrize('manual_read', [True, False])
def test_crafting_image(test_image_uri_doc, manual_read):
    from copy import deepcopy
    doc = Document(test_image_uri_doc, copy=True)
    doc.convert_image_uri_to_blob()
    norm = ImageNormalizer(resize_dim=123,
                           img_mean=(0.1, 0.1, 0.1),
                           img_std=(0.5, 0.5, 0.5))
    img = norm._load_image(doc.blob)
    assert isinstance(img, Image)
    assert img.size == (96, 96)

    img_resized = norm._resize_short(img)
    assert img_resized.size == (123, 123)
    assert isinstance(img_resized, Image)

    norm.resize_dim = (123, 456)
    img_resized = norm._resize_short(img)
    assert img_resized.size == (123, 456)
    assert isinstance(img_resized, Image)

    with pytest.raises(ValueError):
        norm.resize_dim = (1, 2, 3)
        norm._resize_short(img)

    norm.resize_dim = 256
    img = norm._resize_short(img)

    norm.target_size = 128
    cropped_img, b1, b2 = norm._crop_image(img, how='random')
    assert cropped_img.size == (128, 128)
    assert isinstance(cropped_img, Image)

    norm.target_size = 224
    img, b1, b2 = norm._crop_image(img, how='center')
    assert img.size == (224, 224)
    assert isinstance(img, Image)
    assert b1 == 16
    assert b2 == 16

    img = np.asarray(img).astype('float32') / 255

    norm_img = norm._normalize(norm._load_image(doc.blob))

    img -= np.array([[[0.1, 0.1, 0.1]]])
    img /= np.array([[[0.5, 0.5, 0.5]]])

    assert np.array_equal(norm_img, img)

    if manual_read:
        docs = DocumentArray([doc])
    else:
        docs = DocumentArray([test_image_uri_doc])
    processed_docs = norm.craft(docs)
    assert np.array_equal(processed_docs[0].blob, img)


def test_move_channel_axis(test_image_uri_doc):
    norm = ImageNormalizer(channel_axis=2, target_channel_axis=0)

    doc = test_image_uri_doc
    doc.convert_image_uri_to_blob()
    img = norm._load_image(doc.blob)
    assert img.size == (96, 96)

    channel0_img = norm._move_channel_axis(doc.blob, 2, 0)
    assert channel0_img.shape == (3, 96, 96)

    processed_docs = norm.craft(DocumentArray([doc]))
    assert processed_docs[0].blob.shape == (3, 224, 224)
