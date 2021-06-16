FROM jinaai/jina:master

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install -r tests/requirements.txt && pip install .
RUN pytest tests

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]