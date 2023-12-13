FROM tensorflow/tensorflow:2.6.0-gpu

RUN useradd --create-home preprocess

USER preprocess
WORKDIR /home/preprocess
VOLUME /BERT_Manifest

COPY docker_preprocess/req_docker.txt .
RUN python3 -m pip install --user --no-cache-dir -r req_docker.txt

COPY master_list/master_list .
COPY docker_preprocess/manifest_preprocess.py .
COPY taboo_list .
COPY taboo_list_v2 .
COPY xml_tag .
COPY docker_preprocess/preprocess.sh .

