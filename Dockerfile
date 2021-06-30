FROM jinaai/jina:2.0

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install .

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]