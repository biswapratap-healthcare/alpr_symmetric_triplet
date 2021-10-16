import os
import random

import numpy as np
import pandas as pd
from one_hot_map import one_hot_map
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input


num_of_epochs = 5000
path_base = '.\\..\\vsr3pp\\input\\'
path_csv = os.path.join(path_base, 'all_rois.csv')
image_size = 224
batch_size = 1
dimension = 512
vgg16_emb_dim = 25088
text_emb_dim = 370
path_train = os.path.join(path_base, 'all_rois')
vgg16_model = VGG16(weights='imagenet', include_top=False)

data = pd.read_csv(path_csv)
data = data[:21]
train, test = train_test_split(data, train_size=0.7, random_state=1337)
file_id_mapping_train = {k: v for k, v in zip(train.imgID.values, train.GT.values)}
file_id_mapping_test = {k: v for k, v in zip(test.imgID.values, test.GT.values)}
train_len = len(file_id_mapping_train)
test_len = len(file_id_mapping_test)
steps_per_epoch = int(train_len / batch_size)
validation_steps = int(test_len / batch_size)

plate_files_train = list(file_id_mapping_train.keys())
plate_nums_train = list(file_id_mapping_train.values())
o_n_train = len(plate_nums_train)
n_train = len(plate_nums_train) * num_of_epochs

plate_files_test = list(file_id_mapping_test.keys())
plate_nums_test = list(file_id_mapping_test.values())
o_n_test = len(plate_nums_test)
n_test = len(plate_nums_test) * num_of_epochs


def get_feature_vectors(img_path):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    _vgg16_feature = vgg16_model.predict(img_data)
    return _vgg16_feature


def get_text_one_hot_encoding(txt):
    length = len(txt)
    if length < 10:
        num = 10 - length
        for _ in range(0, num, 1):
            txt = txt + '#'
    elif length > 10:
        txt = txt[0:10]
    _encoding = list()
    for t in txt:
        _encoding.extend(one_hot_map.get(t))
    _encoding = np.array(_encoding)
    return _encoding


def generate(is_train=True):
    start = 0
    if is_train:
        o_n = o_n_train
    else:
        o_n = o_n_test
    while True:
        list_anchor_itt = list()
        list_positive_itt = list()
        list_negative_itt = list()

        list_anchor_tii = list()
        list_positive_tii = list()
        list_negative_tii = list()

        if start + batch_size >= o_n:
            # print("Resetting Start ...")
            start = 0

        if is_train:
            plate_nums = plate_nums_train
            plate_files = plate_files_train
        else:
            plate_nums = plate_nums_test
            plate_files = plate_files_test

        anchor_index = start

        negative_indexes = random.sample(range(0, o_n - 1), 5)
        while anchor_index in negative_indexes: negative_indexes.remove(anchor_index)

        # print(anchor_index)
        # print(negative_indexes)

        for negative_index in negative_indexes:
            p = os.path.join(path_train, plate_files[anchor_index])
            p_roi = p + '.roi0.jpg'
            p_troi = p + '.troi0.jpg'
            if os.path.exists(p_roi):
                anchor_image_itt = p_roi
            elif os.path.exists(p_troi):
                anchor_image_itt = p_troi
            else:
                continue
            anchor_image_vector_itt = get_feature_vectors(anchor_image_itt).flatten().tolist()
            positive_text_itt = plate_nums[anchor_index]
            positive_text_vector_itt = get_text_one_hot_encoding(positive_text_itt).tolist()
            negative_text_itt = plate_nums[negative_index]
            negative_text_vector_itt = get_text_one_hot_encoding(negative_text_itt).tolist()

            list_anchor_itt.append(anchor_image_vector_itt)
            list_positive_itt.append(positive_text_vector_itt)
            list_negative_itt.append(negative_text_vector_itt)

            anchor_text_tii = plate_nums[anchor_index]
            anchor_text_vector_tii = get_text_one_hot_encoding(anchor_text_tii).tolist()
            p = os.path.join(path_train, plate_files[anchor_index])
            p_roi = p + '.roi0.jpg'
            p_troi = p + '.troi0.jpg'
            if os.path.exists(p_roi):
                positive_image_tii = p_roi
            elif os.path.exists(p_troi):
                positive_image_tii = p_troi
            else:
                continue
            positive_image_vector_tii = get_feature_vectors(positive_image_tii).flatten().tolist()
            p = os.path.join(path_train, plate_files[negative_index])
            p_roi = p + '.roi0.jpg'
            p_troi = p + '.troi0.jpg'
            if os.path.exists(p_roi):
                negative_image_tii = p_roi
            elif os.path.exists(p_troi):
                negative_image_tii = p_troi
            else:
                continue
            negative_image_vector_tii = get_feature_vectors(negative_image_tii).flatten().tolist()

            list_anchor_tii.append(anchor_text_vector_tii)
            list_positive_tii.append(positive_image_vector_tii)
            list_negative_tii.append(negative_image_vector_tii)

        start += batch_size

        sizes = [len(list_anchor_itt), len(list_positive_itt), len(list_negative_itt),
                 len(list_anchor_tii), len(list_positive_tii), len(list_negative_tii)]
        min_size = min(sizes)

        if min_size == 0:
            continue

        list_anchor_itt = list_anchor_itt[:min_size]
        list_positive_itt = list_positive_itt[:min_size]
        list_negative_itt = list_negative_itt[:min_size]

        list_anchor_tii = list_anchor_tii[:min_size]
        list_positive_tii = list_positive_tii[:min_size]
        list_negative_tii = list_negative_tii[:min_size]

        list_anchor_itt = np.array(list_anchor_itt)
        list_positive_itt = np.array(list_positive_itt)
        list_negative_itt = np.array(list_negative_itt)

        list_anchor_tii = np.array(list_anchor_tii)
        list_positive_tii = np.array(list_positive_tii)
        list_negative_tii = np.array(list_negative_tii)

        yield {'anchor_itt': list_anchor_itt, 'positive_itt': list_positive_itt, 'negative_itt': list_negative_itt,
               'anchor_tii': list_anchor_tii, 'positive_tii': list_positive_tii, 'negative_tii': list_negative_tii},\
               None
