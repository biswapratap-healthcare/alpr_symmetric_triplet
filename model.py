import pandas as pd
from keras import backend as k
from keras import Sequential, Input, Model
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizer_v2.adam import Adam
from sklearn.model_selection import train_test_split

from utils import path_csv, vgg16_emb_dim, text_emb_dim, generate, steps_per_epoch, validation_steps, \
    file_id_mapping_train, file_id_mapping_test


def symmetric_triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    _anchor_itt_emb, _positive_itt_emb, _negative_itt_emb, \
    _anchor_tii_emb, _positive_tii_emb, _negative_tii_emb = inputs
    positive_distance_itt = k.square(_anchor_itt_emb - _positive_itt_emb)
    negative_distance_itt = k.square(_anchor_itt_emb - _negative_itt_emb)
    positive_distance_tii = k.square(_anchor_tii_emb - _positive_tii_emb)
    negative_distance_tii = k.square(_anchor_tii_emb - _negative_tii_emb)
    if dist == 'euclidean':
        positive_distance_itt = k.sqrt(k.sum(positive_distance_itt, axis=-1, keepdims=True))
        negative_distance_itt = k.sqrt(k.sum(negative_distance_itt, axis=-1, keepdims=True))
        positive_distance_tii = k.sqrt(k.sum(positive_distance_tii, axis=-1, keepdims=True))
        negative_distance_tii = k.sqrt(k.sum(negative_distance_tii, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance_itt = k.sum(positive_distance_itt, axis=-1, keepdims=True)
        negative_distance_itt = k.sum(negative_distance_itt, axis=-1, keepdims=True)
        positive_distance_tii = k.sum(positive_distance_tii, axis=-1, keepdims=True)
        negative_distance_tii = k.sum(negative_distance_tii, axis=-1, keepdims=True)
    loss_itt = positive_distance_itt - negative_distance_itt
    loss_tii = positive_distance_tii - negative_distance_tii
    if margin == 'maxplus':
        loss_itt = k.maximum(0.0, 1 + loss_itt)
        loss_tii = k.maximum(0.0, 1 + loss_tii)
    elif margin == 'softplus':
        loss_itt = k.log(1 + k.exp(loss_itt))
        loss_tii = k.log(1 + k.exp(loss_tii))
    loss_itt = k.mean(loss_itt)
    loss_tii = k.mean(loss_tii)
    loss = loss_itt + loss_tii
    return loss


def get_t2nn_model():
    _model = Sequential()
    _model.add(Dense(2048, activation='linear'))
    _model.add(Activation('sigmoid'))
    _model.add(Dense(512, activation='linear'))
    _model.add(BatchNormalization())
    return _model


def get_f2nn_model():
    _model = Sequential()
    _model.add(Dense(2048, activation='linear'))
    _model.add(Activation('sigmoid'))
    _model.add(Dense(512, activation='linear'))
    _model.add(BatchNormalization())
    return _model


def get_models():
    anchor_itt = Input(vgg16_emb_dim, name='anchor_itt')
    positive_itt = Input(text_emb_dim, name='positive_itt')
    negative_itt = Input(text_emb_dim, name='negative_itt')

    _t2nn_embedding_model = get_t2nn_model()
    _f2nn_embedding_model = get_f2nn_model()

    _anchor_itt_emb = _f2nn_embedding_model(anchor_itt)
    _positive_itt_emb = _t2nn_embedding_model(positive_itt)
    _negative_itt_emb = _t2nn_embedding_model(negative_itt)

    anchor_tii = Input(text_emb_dim, name='anchor_tii')
    positive_tii = Input(vgg16_emb_dim, name='positive_tii')
    negative_tii = Input(vgg16_emb_dim, name='negative_tii')

    _anchor_tii_emb = _t2nn_embedding_model(anchor_tii)
    _positive_tii_emb = _f2nn_embedding_model(positive_tii)
    _negative_tii_emb = _f2nn_embedding_model(negative_tii)

    inputs = [anchor_itt, positive_itt, negative_itt,
              anchor_tii, positive_tii, negative_tii]
    outputs = [_anchor_itt_emb, _positive_itt_emb, _negative_itt_emb,
               _anchor_tii_emb, _positive_tii_emb, _negative_tii_emb]

    _triplet_model = Model(inputs, outputs)
    _triplet_model.add_loss(k.mean(symmetric_triplet_loss(outputs)))

    return _triplet_model, _t2nn_embedding_model, _f2nn_embedding_model


if __name__ == "__main__":
    triplet_model, t2nn_embedding_model, f2nn_embedding_model = get_models()

    gen_tr = generate(file_id_mapping_train)
    gen_te = generate(file_id_mapping_test)

    triplet_model.compile(loss=None, optimizer=Adam(0.01))
    history = triplet_model.fit_generator(gen_tr,
                                          validation_data=gen_te,
                                          epochs=10,
                                          verbose=1,
                                          workers=1,
                                          steps_per_epoch=steps_per_epoch,
                                          validation_steps=validation_steps,
                                          use_multiprocessing=False)

    t2nn_embedding_model.save_weights(filepath='t2nn.h5')
    f2nn_embedding_model.save_weights(filepath='f2nn.h5')
