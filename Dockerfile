FROM jinaai/jina:master

ADD *.py *.yml requirements.txt ./

RUN pip install -r tests/requirements.txt

RUN pytest tests

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]