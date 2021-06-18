import os

from jina import Flow, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(
            uri=os.path.join(cur_dir, '..', 'data', 'test_image.png'))
        doc.convert_image_uri_to_blob()
        yield doc


def test_use_in_flow():
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(on='/index', inputs=data_generator(5))
        docs = data[0].docs
        for doc in docs:
            assert doc.blob.shape == (64, 64, 3)
