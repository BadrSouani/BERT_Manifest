THE GITHUB IS IN PROGRESS


This project fine-tune with APK's manifests a BERT model pre-trained on wikipedia to identify malwares.

In order to launch the AI, the project is formed of 3 different docker containers.
The first one in 'docker_manifestGetter' take the manifests from APKs of Androzoo.
The second (Optional) 'docker_preprocess' is in charge of the pre-processing if needed.
And 'docker_bert' is the BERT model ready to learn. 


First, create the 'manifest/' directory into the main folder. Then go into 'docker_manifestGetter' directory,
and change 'APIKEY' in 'manifest_getter.py'  by your own key giving you access to Androzoo APKs.
Go back into the parent directory and build the docker image using the 'Dockerfile' in that folder with
>>  docker build -t manifestgetter -f docker_manifestGetter/Dockerfile .  <<.
Now that you created the image, run it with
>>  docker run --name=manifestgetter -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest manifestgetter /bin/bash  <<.
Enter the docker container 'manifestgetter' and execute
>>  python manifest_getter.py master_list  <<.
'master_list' contains the sha and target of the APKs that are going to be used. 
Once the process is finished, the previously empty directory '/manifest' should have the manifests at '/manifest/manifest'
in your container, and in your 'manifest/manifest' folder in the project from your host machine.


Now let's create our preprocessed manifests. From the project, build the next image with
>> docker build -t preprocess -f docker_preprocess/Dockerfile . <<.
Run it with
>> docker run --name=preprocess -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest preprocess /bin/bash <<.
You can find in the container the file 'taboo_list' which is a list of words to delete from the manifests,
and 'xml_tag' the key words used in order to keep (or delete) specific lines.
In the container, run
>> python manifest_preprocess.py --help << to get an idea of what you can do.
(e.g >> python manifest_preprocess.py --master_list=master_list --taboo_list=taboo_list --xml_tag=xml_tag /manifest/preprocess/ N << with N the number of manifests.)
Preprocessed manifests should then be found in the path given as argument. 


Last but not least, in order to create the bert model
>> docker build -t bert -f docker_bert/Dockerfile . <<.
Then
>> docker run --name=bert -it --mount type=bind,src="$(pwd)"/manifest,dst=/manifest bert /bin/bash <<.
As before,
>> python BERT_Manifest_TF.py --help <<
to get more informations.
The 'master_list' folder contains ten different list, with the same 265000 lines each but in a different order. 
The python file  has almost the same arguments as the preprocessor since you can alos generate them directly from this file without saving them.
(e.g >> python BERT_Manifest_TF.py N 0.5 0.2 10 --master_list=master_list/master_list --path=/manifest/preprocess/  --comment=ex <<)
You will then find a 'trained_models' folder with the generated bert model, a csv file with the results and a graphic representation. 
