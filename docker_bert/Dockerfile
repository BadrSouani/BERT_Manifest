FROM tensorflow/tensorflow:2.8.0-gpu

RUN useradd --create-home bert

USER bert
WORKDIR /home/bert
VOLUME /manifest
VOLUME /output

RUN curl -L  'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3?tf-hub-format=compressed' -o bert_en_uncased_preprocess_3.tgz
RUN mkdir bert_en_uncased_preprocess_3
RUN tar xvzf bert_en_uncased_preprocess_3.tgz -C bert_en_uncased_preprocess_3
RUN rm bert_en_uncased_preprocess_3.tgz
RUN curl -L 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4?tf-hub-format=compressed' -o bert_en_uncased_L-12_H-768_A-12_4.tgz
RUN mkdir bert_en_uncased_L-12_H-768_A-12_4
RUN tar xvzf bert_en_uncased_L-12_H-768_A-12_4.tgz -C bert_en_uncased_L-12_H-768_A-12_4
RUN rm bert_en_uncased_L-12_H-768_A-12_4.tgz

COPY docker_bert/req_docker.txt .
RUN python3 -m pip install --user --no-cache-dir -r req_docker.txt
RUN mkdir output

COPY families/ families/
COPY master_list/ master_list/
COPY docker_bert/BERT_Manifest_TF.py .
COPY docker_bert/BERT_Manifest_TF_multiclass.py .
COPY taboo_list .
COPY xml_tag .
COPY docker_bert/test_bert.sh .
