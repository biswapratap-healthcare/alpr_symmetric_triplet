import os
import random
import numpy as np
from one_hot_map import one_hot_map
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


path_base = '.\\..\\vsr3pp\\input\\'
path_csv = os.path.join(path_base, 'all_rois.csv')
image_size = 224
batch_size = 24
vgg16_emb_dim = 25088
text_emb_dim = 370
path_train = os.path.join(path_base, 'all_rois')
vgg16_model = VGG16(weights='imagenet', include_top=False)


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


def generate(file_id_mapping):
    index = 1
    n = len(file_id_mapping)
    plate_files = list(file_id_mapping.keys())
    plate_nums = list(file_id_mapping.values())
    while index < n:
        list_anchor_itt = list()
        list_positive_itt = list()
        list_negative_itt = list()
        list_anchor_tii = list()
        list_positive_tii = list()
        list_negative_tii = list()

        for i in range(batch_size):
            idx_itt = index
            anchor_image_itt = os.path.join(path_train, plate_files[idx_itt])
            if os.path.exists(anchor_image_itt + '.roi0.jpg'):
                anchor_image_itt = anchor_image_itt + '.roi0.jpg'
            elif os.path.exists(anchor_image_itt + '.troi0.jpg'):
                anchor_image_itt = anchor_image_itt + '.troi0.jpg'
            else:
                continue
            anchor_image_vector_itt = get_feature_vectors(anchor_image_itt).flatten().tolist()
            positive_text_itt = plate_nums[idx_itt]
            positive_text_vector_itt = get_text_one_hot_encoding(positive_text_itt).tolist()
            negative_text_itt = plate_nums[idx_itt - 1]
            negative_text_vector_itt = get_text_one_hot_encoding(negative_text_itt).tolist()

            list_anchor_itt.append(anchor_image_vector_itt)
            list_positive_itt.append(positive_text_vector_itt)
            list_negative_itt.append(negative_text_vector_itt)

            idx_tii = index
            anchor_text_tii = plate_nums[idx_tii]
            anchor_text_vector_tii = get_text_one_hot_encoding(anchor_text_tii).tolist()
            positive_image_tii = os.path.join(path_train, plate_files[idx_tii])
            if os.path.exists(positive_image_tii + '.roi0.jpg'):
                positive_image_tii = positive_image_tii + '.roi0.jpg'
            elif os.path.exists(positive_image_tii + '.troi0.jpg'):
                positive_image_tii = positive_image_tii + '.troi0.jpg'
            else:
                continue
            positive_image_vector_tii = get_feature_vectors(positive_image_tii).flatten().tolist()
            negative_image_tii = os.path.join(path_train, plate_files[idx_tii - 1])
            if os.path.exists(negative_image_tii + '.roi0.jpg'):
                negative_image_tii = negative_image_tii + '.roi0.jpg'
            elif os.path.exists(negative_image_tii + '.troi0.jpg'):
                negative_image_tii = negative_image_tii + '.troi0.jpg'
            else:
                continue
            negative_image_vector_tii = get_feature_vectors(negative_image_tii).flatten().tolist()

            list_anchor_itt.append(anchor_image_vector_itt)
            list_positive_itt.append(positive_text_vector_itt)
            list_negative_itt.append(negative_text_vector_itt)

            list_anchor_tii.append(anchor_text_vector_tii)
            list_positive_tii.append(positive_image_vector_tii)
            list_negative_tii.append(negative_image_vector_tii)

        list_anchor_itt = np.array(list_anchor_itt)
        list_positive_itt = np.array(list_positive_itt)
        list_negative_itt = np.array(list_negative_itt)

        list_anchor_tii = np.array(list_anchor_tii)
        list_positive_tii = np.array(list_positive_tii)
        list_negative_tii = np.array(list_negative_tii)

        yield {'anchor_itt': list_anchor_itt, 'positive_itt': list_positive_itt, 'negative_itt': list_negative_itt,
               'anchor_tii': list_anchor_tii, 'positive_tii': list_positive_tii, 'negative_tii': list_negative_tii},\
               None
