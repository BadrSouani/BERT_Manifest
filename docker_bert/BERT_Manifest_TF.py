import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import tensorflow_addons as tfa
import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import csv
#from shutil import copy
from sklearn.model_selection import train_test_split

def dir_path(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)

def getFStr(x):
    return str(round(x, 5))



parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('--model', type=str, default="", help="Path to the model to reload")
parser.add_argument('--master_list', type=str, default="master_list/master_list", help="Path of the Manifest list")
parser.add_argument('--path', type=dir_path, default="manifest/", help="Path of the Manifest directory")
parser.add_argument('--output_dir', type=dir_path, default="/output/", help="Output Directory")
parser.add_argument("N", help="Number of Application to use", type=int)
parser.add_argument("train", help="Percentage of N to use for training 0>train>1", type=float)
parser.add_argument("valid", help="Percentage of N to use for validation 0>train>1", type=float)
parser.add_argument("epoch", help="Number of epochs", type=int)
parser.add_argument("--use_weight", type=bool, default=False, help="By default do not change weights, use flag if want to use weights modifications")
parser.add_argument("--weight1", default=0.5, type=float,
                    help="Percentage of the total weight represented by positive manifests")
parser.add_argument("--seed", help="Shuffling before split, same int for same shuffle", type=int, default=None)
parser.add_argument("--stratify", type=bool, default=False, help="By default do not stratify sets, use flag if want to stratify train/val/test sets")
parser.add_argument("--ablation", default=False,
                    help="0 (default) if xml_tag list contains tags to keep, else delete them from manifests",
                    type=bool)
parser.add_argument("--xml_tag", default="",
                    help="Path to file with tags to keep or delete before application of the taboo_list", type=str)
parser.add_argument("--taboo_list", nargs='?', default="", help='Path to taboo list for manifests', type=str)
parser.add_argument("--comment", nargs='?', default="", help="Added to result folder's name ", type=str)
args = parser.parse_args()

cpt_to_load = args.model
nApp = args.N
f_train = args.train
f_valid = args.valid
epochs = args.epoch
use_weight = args.use_weight
percentage_weight_1 = args.weight1
seed = args.seed
stratify = args.stratify
comment = args.comment
ablation = args.ablation
MANIFESTS_DIR = args.path
TABOO_DIR = args.taboo_list
TAGS_DIR = args.xml_tag
master_list_name = args.master_list
OUTPUT_DIR = args.output_dir
if not os.path.isdir(OUTPUT_DIR):
    raise(NotADirectoryError(OUTPUT_DIR))

CLASS_NUMBER=2
batch_size = 32
init_lr = 3e-5

now = datetime.now()
dataset_name = 'trained_models/manifest_'+comment+'_['+ str(nApp) + '_' + str(f_train) + '_' + str(f_valid) + '_' +str(epochs) + ']'

preprocessor = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

#preprocessor = '../bert_en_uncased_preprocess_3'
#bert = '../bert_en_uncased_L-12_H-768_A-12_4'

master_list = pd.read_csv(master_list_name, header=None).to_numpy()
truncated_array = master_list[:nApp]

#Manifest preprocess #######
# No tag selection
if "" == TAGS_DIR:
    for i in range(nApp):
        f_in = open(MANIFESTS_DIR + truncated_array[i][0], encoding='utf-8')
        truncated_array[i][0] = f_in.read()
        f_in.close()

# Tag selection
else:
    manifest_content = {}
    tag_names = open(TAGS_DIR, 'r').readlines()
    len_tag = len(tag_names)
    for i in range(len_tag):
        tag_names[i] = tag_names[i].replace('\n', '')
    # Per manifest
    for i in range(nApp):
        f_in = open(MANIFESTS_DIR + truncated_array[i][0], encoding='utf-8')
        manifest_content[i] = f_in.readlines()
        f_in.close()
        manifest_filtered = ''
        # per line
        for line in manifest_content[i]:
            keep = ablation
            counter = 0
            while ((keep == 0 and ablation == 0) or (keep and ablation)) and counter < len_tag:
                if tag_names[counter] in line : keep = 1-ablation
                counter += 1
            if keep : manifest_filtered += line
        truncated_array[i][0] = manifest_filtered

# Taboo list usage
if not "" == TABOO_DIR:
    taboo = open(TABOO_DIR, 'r').readlines()
    for i in range(len(taboo)):
        taboo[i] = taboo[i].replace('\n', '')
    for i in range(nApp):
        for word in taboo:
            truncated_array[i][0] = truncated_array[i][0].replace(word, '')

##############################
#Dataset split ###############
xml = np.asarray(truncated_array[:, 0])
target = np.asarray(truncated_array[:, 1]).astype('int32')

f_test = 1-f_train-f_valid
xml_remaining, xml_test, target_remaining, target_test = train_test_split(xml, target, test_size=f_test, stratify=target if stratify else None, random_state=seed, shuffle= True if stratify else False )

ratio_remaining = 1 - f_test
ratio_val_adjusted = f_valid/ratio_remaining
xml_train, xml_val, target_train, target_val = train_test_split(xml_remaining, target_remaining, test_size=ratio_val_adjusted, stratify=target_remaining if stratify else None, random_state=seed , shuffle= True if stratify else False)

target_train_categorical = tf.keras.utils.to_categorical(target_train, num_classes=CLASS_NUMBER, dtype="int32")
target_validation_categorical = tf.keras.utils.to_categorical(target_val, num_classes=CLASS_NUMBER, dtype="int32")
target_test_categorical = tf.keras.utils.to_categorical(target_test, num_classes=CLASS_NUMBER, dtype="int32")

train_ds = tf.data.Dataset.from_tensor_slices((xml_train, target_train_categorical))
valid_ds = tf.data.Dataset.from_tensor_slices((xml_val, target_validation_categorical))
test_ds = tf.data.Dataset.from_tensor_slices((xml_test, target_test_categorical))

# Weight definition
#Number of each family
train_0 = len(list(train_ds.filter(lambda x,y : y[0]==1)))
train_1 = xml_train.size-train_0
val_0 = len(list(valid_ds.filter(lambda x,y : y[0]==1)))
val_1 = xml_val.size-val_0
test_0 = len(list(test_ds.filter(lambda x,y : y[0]==1)))
test_1 = xml_test.size-test_0

train_string = "Train \t"+str(xml_train.size)+":["+str(train_0)+"/"+str(train_1)+"]"
val_string = "Validation \t"+str(xml_val.size)+":["+str(val_0)+"/"+str(val_1)+"]"
test_string = "Test \t"+str(xml_test.size)+":["+str(test_0)+"/"+str(test_1)+"]"

print("Repartition of split with Total:[Neg/Pos]")
print(train_string)
print(val_string)
print(test_string)
#percentage of weight of each family from the total weight
percentage_weight_0 = 1 - percentage_weight_1

#weight for a sample of each family
weight0 = 1.
weight1 = (percentage_weight_1 * train_0 * weight0) / (train_1 * (1 - percentage_weight_1))

class_weight = {0: weight0, 1: weight1}
weight_string = (("Used" if use_weight == 1 else "Not used") + " weights Neg/Pos :"+str(class_weight))
print(weight_string)

train_ds = train_ds.batch(batch_size=batch_size)
valid_ds = valid_ds.batch(batch_size=batch_size)
test_ds = test_ds.batch(batch_size=batch_size)

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocessor, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(128, activation='relu', name='classifier128')(net)
    net = tf.keras.layers.Dense(64, activation='relu', name='classifier64')(net)
    net = tf.keras.layers.Dense(CLASS_NUMBER, activation='softmax', name='classifier1')(net)
    return tf.keras.Model(text_input, net)

saved_model_path = os.path.join(OUTPUT_DIR, dataset_name)

checkpoint_filepath = saved_model_path + '/checkpoint/best_val_loss_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

if cpt_to_load != "" : classifier_model = tf.keras.models.load_model(cpt_to_load)
else : classifier_model = build_classifier_model()

metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(class_id=1), tf.keras.metrics.Recall(class_id=1), tfa.metrics.F1Score(num_classes=CLASS_NUMBER)]


classifier_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics,
                         optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))

#with tf.device('/CPU:0'):
start_time = time.time()
print(f'Training model with {bert}')
history = classifier_model.fit(x=train_ds,
                               validation_data=valid_ds,
                               epochs=epochs,
                               class_weight=class_weight if use_weight == 1 else None,
                               callbacks=[model_checkpoint_callback])
training_time=time.time() - start_time;
test_results= classifier_model.evaluate(test_ds)


print(f'Loss: {test_results[0]}')
print(f'Accuracy: {test_results[1]}')
print(f"--- {getFStr(training_time)} seconds ---\n")


if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
classifier_model.save(saved_model_path)

if epochs>0 :
    history_dict = history.history
    acc = history_dict['categorical_accuracy']
    lo = history_dict['loss']
    val_acc = history_dict['val_categorical_accuracy']
    val_lo = history_dict['val_loss']
    dict_arrays = list(history_dict.values())
    dict_size = len(dict_arrays)
    range_dict = range(dict_size)

epochs_r = range(1, epochs + 1)

output_basename = os.path.join(OUTPUT_DIR, dataset_name, f"{str(nApp)}_-{comment}")

csv_file = output_basename + ".csv"

csvfile = open(csv_file, 'w', newline='', encoding='utf-8')
csv_header = history.model.metrics_names
csv_header.insert(0,"epochs")
writer = csv.writer(csvfile, delimiter=';')
writer.writerow([str(args), getFStr(training_time)])#ligne de code, temps, test acc etc
writer.writerow(csv_header)
#ecrire les lignes

for i in range(epochs):
    temp_list=[dict_arrays[x][i] for x in range_dict]
    temp_list.insert(0, i+1)
    writer.writerow(temp_list)
writer.writerow([])
test_results.insert(0,"test")
writer.writerow(test_results)
writer.writerow([])
writer.writerow([class_weight])
writer.writerow([train_string])
writer.writerow([val_string])
writer.writerow([test_string])
csvfile.close()
