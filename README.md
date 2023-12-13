# Malware Detection using Fine-Tuned BERT Model

## Overview

This project utilizes a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model, pre-trained on Wikipedia, to identify malware in Android application packages (APKs). The process involves extracting information from the APK manifests using Docker containers and applying pre-processing steps if necessary. The final step is training the BERT model on the preprocessed data.

## Docker Containers

1. **docker_manifestGetter:**
   - Extracts manifests from APKs using Androzoo.
   - Create a 'manifest/' directory in the main folder.
   - Update 'APIKEY' in 'docker_manifestGetter/manifest_getter.py' with your Androzoo API key.
   - Build the Docker image:
     ```
     docker build -t manifestgetter -f docker_manifestGetter/Dockerfile .
     ```
   - Run the Docker container:
     ```
     docker run --name=manifestgetter -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest manifestgetter /bin/bash
     ```
   - Inside the container, execute:
     ```
     python manifest_getter.py master_list
     ```
   - Manifests will be available at '/manifest/manifest' in the container and in 'manifest/manifest' on your host machine.

2. **docker_preprocess (Optional):**
   - Performs pre-processing on manifests if needed.
   - Build the Docker image:
     ```
     docker build -t preprocess -f docker_preprocess/Dockerfile .
     ```
   - Run the Docker container:
     ```
     docker run --name=preprocess -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest preprocess /bin/bash
     ```
   - Inside the container, run:
     ```
     python manifest_preprocess.py --help
     ```
   - Customize pre-processing options, e.g.:
     ```
     python manifest_preprocess.py --master_list=master_list --taboo_list=taboo_list --xml_tag=xml_tag /manifest/preprocess/ N
     ```
   - Preprocessed manifests will be available at the specified path.

3. **docker_bert:**
   - Creates and trains the BERT model.
   - Build the Docker image:
     ```
     docker build -t bert -f docker_bert/Dockerfile .
     ```
   - Run the Docker container:
     ```
     docker run --name=bert -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest bert /bin/bash
     ```
   - Inside the container, run:
     ```
     python BERT_Manifest_TF.py --help
     ```
   - Configure BERT model training, e.g.:
     ```
     python BERT_Manifest_TF.py N 0.5 0.2 10 --master_list=master_list/master_list --path=/manifest/preprocess/ --comment=ex
     ```
   - Find the trained model in the 'trained_models' folder, along with a CSV file of results and a graphical representation.

## Note
- Ensure you have the necessary permissions and dependencies installed to run Docker containers.
- Modify parameters as needed for your specific use case.
- Refer to the respective help commands for more detailed options.
