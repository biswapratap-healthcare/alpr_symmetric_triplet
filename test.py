import os
import faiss
import numpy as np
from tensorflow import keras
from utils import dimension, get_text_one_hot_encoding, plate_nums_train, plate_files_train, path_train, \
    get_feature_vectors

if __name__ == "__main__":
    f2nn_embedding_model = keras.models.load_model('f2nn.h5')
    t2nn_embedding_model = keras.models.load_model('t2nn.h5')

    if os.path.exists('vector.index'):
        x = list()
        index = faiss.read_index("vector.index")
        plate_file = plate_files_train[0]
        plate_file = os.path.join(path_train, plate_file)
        plate_file = plate_file + '.roi0.jpg'
        print(plate_file)
        plate_vector = get_feature_vectors(plate_file).flatten().tolist()
        x.append(plate_vector)
        x = np.array(x)
        image_embedding = f2nn_embedding_model.predict(x=x).flatten().tolist()
        search_vector = list()
        search_vector.append(image_embedding)
        search_vector = np.array(search_vector)
        search_vector = np.float32(search_vector)
        distances, indices = index.search(search_vector, 3)
        print(distances)
        print(indices)
        indices = indices.flatten().tolist()
        print(plate_nums_train[indices[0]])
    else:
        db_vectors = list()
        index = 0
        for plate_num in plate_nums_train[:11]:
            x = list()
            k_vec = get_text_one_hot_encoding(plate_num).tolist()
            x.append(k_vec)
            x = np.array(x)
            text_embedding = t2nn_embedding_model.predict(x=x).flatten().tolist()

            db_vectors.append(text_embedding)
            if index % 100 == 0:
                print(index)
            index += 1

        db_vectors = np.array(db_vectors)
        db_vectors = np.float32(db_vectors)

        nlist = 2  # number of clusters
        quantiser = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)

        print(index.is_trained)
        index.train(db_vectors)  # train on the database vectors
        print(index.ntotal)
        index.add(db_vectors)  # add the vectors and update the index
        print(index.is_trained)
        print(index.ntotal)

        faiss.write_index(index, "vector.index")
