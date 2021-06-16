FROM jinaai/jina:master as base

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest tests

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]