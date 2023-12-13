FROM tensorflow/tensorflow:2.6.0-gpu

RUN useradd --create-home manifestGetter

USER manifestGetter
WORKDIR /home/manifestGetter
VOLUME /BERT_Manifest

COPY docker_manifestGetter/req_docker.txt .
RUN python3 -m pip install --user --no-cache-dir -r req_docker.txt


COPY master_list/master_list .
COPY docker_manifestGetter/manifest_getter.py .
