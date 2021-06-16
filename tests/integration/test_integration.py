import pytest
import pathlib
import os

from jina import Flow, Document
from jinahub.image.normalizer import ImageNormalizer


@pytest.fixture
def root_dir() -> str:
    return str(pathlib.Path(__file__).parent.parent)


def data_generator(num_docs, root_dir):
    for i in range(num_docs):
        doc = Document(uri=os.path.join(root_dir, 'data', 'test_image.png'))
        doc.convert_image_uri_to_blob()
        yield doc


def test_use_in_flow(root_dir):
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(on='/index', inputs=data_generator(5, root_dir))
        docs = data[0].docs
        for doc in docs:
            assert doc.blob.shape == (64, 64, 3)
