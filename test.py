import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils import get_feature_vectors, path_csv, get_text_one_hot_encoding, path_train


if __name__ == "__main__":
    f2nn_embedding_model = keras.models.load_model('f2nn.h5')
    t2nn_embedding_model = keras.models.load_model('t2nn.h5')

    data = pd.read_csv(path_csv)
    train, test = train_test_split(data, train_size=0.7, random_state=1337)
    file_id_mapping_train = {k: v for k, v in zip(train.imgID.values, train.GT.values)}
    file_id_mapping_test = {k: v for k, v in zip(test.imgID.values, test.GT.values)}

    for k, v in file_id_mapping_test.items():
        p = os.path.join(path_train, k)
        if os.path.exists(p + '.roi0.jpg'):
            p = p + '.roi0.jpg'
        elif os.path.exists(p + '.troi0.jpg'):
            p = p + '.troi0.jpg'
        else:
            continue

        x = list()
        k_vec = get_feature_vectors(p).flatten().tolist()
        x.append(k_vec)
        x = np.array(x)
        image_embedding = f2nn_embedding_model.predict(x=x).flatten().tolist()

        x = list()
        v_vec = get_text_one_hot_encoding(v).tolist()
        x.append(v_vec)
        x = np.array(x)
        text_embedding = t2nn_embedding_model.predict(x=x).flatten().tolist()

