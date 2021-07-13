FROM jinaai/jina:2.0.7-py37-perf

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install .

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
