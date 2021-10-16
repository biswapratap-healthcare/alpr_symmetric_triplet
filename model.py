from keras import Sequential, Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot

from utils import vgg16_emb_dim, text_emb_dim, steps_per_epoch, \
    validation_steps, generate, num_of_epochs

import tensorflow as tf


def squared_dist(a, b):
    assert a.shape.as_list() == b.shape.as_list()
    row_norms_a = tf.reduce_sum(tf.square(a), axis=1)
    row_norms_a = tf.reshape(row_norms_a, [-1, 1])
    row_norms_b = tf.reduce_sum(tf.square(b), axis=1)
    row_norms_b = tf.reshape(row_norms_b, [1, -1])
    return row_norms_a - 2 * tf.matmul(a, tf.transpose(b)) + row_norms_b


def triplet_loss(output_a, output_p, output_n, margin=2):
    euclidean_distance_ap = squared_dist(output_a, output_p)
    euclidean_distance_an = squared_dist(output_a, output_n)
    t = tf.abs(margin + tf.pow(euclidean_distance_ap, 2) - tf.pow(euclidean_distance_an, 2))
    m = tf.reduce_max(t)
    loss_triplet = tf.clip_by_value(t,
                                    clip_value_max=m,
                                    clip_value_min=0.0)
    return loss_triplet


def symmetric_triplet_loss(inputs):
    img_op0, txt_op0, txt_op1, txt_op0, img_op0, img_op1 = inputs
    loss_itt = triplet_loss(img_op0, txt_op0, txt_op1)
    loss_tii = triplet_loss(txt_op0, img_op0, img_op1)
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
    _triplet_model.add_loss(symmetric_triplet_loss(outputs))

    return _triplet_model, _t2nn_embedding_model, _f2nn_embedding_model


if __name__ == "__main__":
    triplet_model, t2nn_embedding_model, f2nn_embedding_model = get_models()

    gen_tr = generate(is_train=True)
    gen_te = generate(is_train=False)

    triplet_model.compile(optimizer=Adam(0.01))
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    history = triplet_model.fit(gen_tr,
                                validation_data=gen_te,
                                epochs=num_of_epochs,
                                verbose=1,
                                workers=1,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_steps,
                                use_multiprocessing=False,
                                callbacks=[es])
    t2nn_embedding_model.save(filepath='t2nn.h5')
    f2nn_embedding_model.save(filepath='f2nn.h5')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
