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
parser.add_argument("--loss", nargs='?', default="categorical_crossentropy", help="Loss function to use during training", type=str)
parser.add_argument("--seed", help="Shuffling before split, same int for same shuffle", type=int, default=None)
parser.add_argument("--comment", nargs='?', default="", help="Added to result folder's name ", type=str)

args = parser.parse_args()
cpt_to_load = args.model
nApp = args.N
f_train = args.train
f_valid = args.valid
epochs = args.epoch
lossfunction = args.loss
seed = args.seed
comment = args.comment
MANIFESTS_DIR = args.path
master_list_name = args.master_list
OUTPUT_DIR = args.output_dir
if not os.path.isdir(OUTPUT_DIR):
    raise (NotADirectoryError(OUTPUT_DIR))

batch_size = 32
init_lr = 3e-5

now = datetime.now()
current_date = str(now.year) + "-" + str(now.month) + "-" + str(now.day)
dataset_name = 'trained_models/manifest_[' + current_date + "]_[" + str(nApp) + '_' + str(f_train) + '_' + str(
    f_valid) + '_' + str(epochs) + "]_" + comment

preprocessor = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

#preprocessor = '../bert_en_uncased_preprocess_3'
#bert = '../bert_en_uncased_L-12_H-768_A-12_4'

master_list = pd.read_csv(master_list_name, header=None).to_numpy()
truncated_array = master_list[:nApp]

for i in range(nApp):
    f_in = open(MANIFESTS_DIR + truncated_array[i][0], encoding='utf-8')
    truncated_array[i][0] = f_in.read()
    f_in.close()

CLASS_NAMES = ['0', '1', '2', '3']
CLASS_NUMBER = len(CLASS_NAMES)

##############################
# Dataset split ###############
xml = np.asarray(truncated_array[:, 0])
target = np.asarray(truncated_array[:, 1]).astype('int32')

f_test = 1 - f_train - f_valid
xml_remaining, xml_test, target_remaining, target_test = train_test_split(xml, target, test_size=f_test,
                                                                          stratify=target, random_state=seed)

ratio_remaining = 1 - f_test
ratio_val_adjusted = f_valid / ratio_remaining
xml_train, xml_val, target_train, target_val = train_test_split(xml_remaining, target_remaining,
                                                                test_size=ratio_val_adjusted, stratify=target_remaining,
                                                                random_state=seed)

target_train_categorical = tf.keras.utils.to_categorical(target_train, num_classes=CLASS_NUMBER, dtype="int32")
target_validation_categorical = tf.keras.utils.to_categorical(target_val, num_classes=CLASS_NUMBER, dtype="int32")
target_test_categorical = tf.keras.utils.to_categorical(target_test, num_classes=CLASS_NUMBER, dtype="int32")

train_ds = tf.data.Dataset.from_tensor_slices((xml_train, target_train_categorical))
valid_ds = tf.data.Dataset.from_tensor_slices((xml_val, target_validation_categorical))
test_ds = tf.data.Dataset.from_tensor_slices((xml_test, target_test_categorical))

zero = len(list(train_ds.filter(lambda x,y : y[0]==1)))
one = len(list(train_ds.filter(lambda x,y : y[1]==1)))
two = len(list(train_ds.filter(lambda x,y : y[2]==1)))
three = len(list(train_ds.filter(lambda x,y : y[3]==1)))

print("Number of Pos samples : 0 :" + str(zero) + ", 1 :" + str(one) + ", 2 :" + str(two) + ", 3 :" + str(three))

# percentage of weight of each family from the total weight
# percentage_weight_0 = 1. / CLASS_NUMBER
# percentage_weight_1 = percentage_weight_0
#
# # weight for a sample of each family
# total_weight = xml_train.size
# weightPerClass = total_weight / CLASS_NUMBER

# weight0 = weightPerClass / zero
# weight1 = weightPerClass / one
# weight2 = weightPerClass / two
# weight3 = weightPerClass / three
#
# class_weight = {0: weight0, 1: weight1, 2: weight2, 3: weight3}
class_weight = None
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
    net = tf.keras.layers.Dense(16, activation='relu', name='classifier16')(net)
    net = tf.keras.layers.Dense(CLASS_NUMBER, activation='softmax', name='classifier1')(net)
    return tf.keras.Model(text_input, net)

saved_model_path = os.path.join(OUTPUT_DIR, dataset_name)
checkpoint_filepath = saved_model_path + '/checkpoint/best_val_loss_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

classifier_model = build_classifier_model()
metrics = [tf.keras.metrics.CategoricalAccuracy(), tfa.metrics.F1Score(num_classes=CLASS_NUMBER)]

classifier_model.compile(loss=lossfunction, metrics=metrics,
                         optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))

#with tf.device('/CPU:0'):
start_time = time.time()
print(f'Training model with {bert}')
history = classifier_model.fit(x=train_ds,
                                validation_data=valid_ds,
                                epochs=epochs,
                                class_weight=class_weight,
                               callbacks=[model_checkpoint_callback])
training_time = time.time() - start_time;

test_results = classifier_model.evaluate(test_ds)

print(f'Loss: {test_results[0]}')
print(f'Accuracy: {test_results[1]}')
print(f"--- {getFStr(training_time)} seconds ---\n")

saved_model_path = os.path.join(OUTPUT_DIR, dataset_name)
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
classifier_model.save(saved_model_path)

if epochs>0:
    history_dict = history.history
    acc = history_dict['categorical_accuracy']
    lo = history_dict['loss']
    val_acc = history_dict['val_categorical_accuracy']
    val_lo = history_dict['val_loss']
    epochs_r = range(1, epochs + 1)
    dict_arrays = list(history_dict.values())
    dict_size = len(dict_arrays)
    range_dict = range(dict_size)

output_basename = os.path.join(OUTPUT_DIR, dataset_name, f"{str(nApp)}_{comment}")

csv_file = output_basename + ".csv"

csvfile = open(csv_file, 'w', newline='', encoding='utf-8')
csv_header = history.model.metrics_names
csv_header.insert(0, "epochs")
writer = csv.writer(csvfile, delimiter=';')
writer.writerow([str(args), current_date, getFStr(training_time)])
writer.writerow(csv_header)

for i in range(epochs):
    temp_list=[dict_arrays[x][i] for x in range_dict]
    temp_list.insert(0, i+1)
    writer.writerow(temp_list)
writer.writerow([])
test_results.insert(0,"test")
writer.writerow(test_results)
writer.writerow([])
writer.writerow([class_weight])
csvfile.close()
