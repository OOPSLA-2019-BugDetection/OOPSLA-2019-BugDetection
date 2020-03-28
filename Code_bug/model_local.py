import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend import ones_like
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.optimizers import adam
from keras.layers import Conv1D, Dense, Reshape, Concatenate, Flatten, Activation, Dropout
from keras.models import Input, Model
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention
from adding_weight import adding_weight
from configparser import ConfigParser
import numpy as np
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


def local_context_learning(input_length, input_dim, output_dim, hidden_dim, filters_num, kernel_val, learning_rate, drop_rate):
    basic_input = Input(shape=(input_length, input_dim))
    label_input = Input(shape=(1,))
    weighted_input = adding_weight(input_length, input_dim)([basic_input, label_input])
    rnn_output = GRU(units=hidden_dim, return_sequences=True)(weighted_input)
    rnn_att = SeqSelfAttention(attention_activation='sigmoid')(rnn_output)
    cnn_output = Conv1D(filters=filters_num, kernel_size=kernel_val, padding="same")(weighted_input)
    cnn_output_reformat = Dense(hidden_dim)(cnn_output)
    cnn_att = SeqSelfAttention(attention_activation='sigmoid')(cnn_output_reformat)
    new_value = Concatenate(axis=1)([rnn_att, cnn_att])
    new_keys = Lambda(lambda x: ones_like(x))(new_value)
    new_result = MultiHeadAttention(head_num=2)([weighted_input, new_keys, new_value])
    result = Flatten()(new_result)
    result_fix = Dropout(rate=drop_rate)(result)
    output = Dense(output_dim)(result_fix)
    fixed_output = Activation(activation='sigmoid')(output)
    model = Model([basic_input, label_input], fixed_output)
    ada = adam(lr=learning_rate)
    model.compile(optimizer=ada, loss='categorical_crossentropy')
    return model


def getting_data(file_path, print_text):
    print("Loading " + print_text + " Data...")
    data = np.load(file_path)
    print("Done")
    return data


cfg = ConfigParser()
cfg.read('config.ini')
input_l = int(cfg.get('localcontext', 'input_length'))
input_d = int(cfg.get('localcontext', 'input_dim'))
output_d = int(cfg.get('localcontext', 'output_dim'))
hidden_d = int(cfg.get('localcontext', 'hidden_dim'))
filters_n = int(cfg.get('localcontext', 'filters_number'))
kernel_v = int(cfg.get('localcontext', 'kernel_value'))
batch_size_num = int(cfg.get('localcontext', 'batch_size_num'))
epoch_num = int(cfg.get('localcontext', 'epoch_num'))
learning_r = float(cfg.get('localcontext', 'learning_rate'))
drop_r = float(cfg.get('localcontext', 'dropout_rate'))
training_input_data = cfg.get('localcontext', 't_input_data')
training_label_data = cfg.get('localcontext', 't_label_data')
training_output_data = cfg.get('localcontext', 't_output_data')
testing_input_data = cfg.get('localcontext', 'test_input_data')
testing_label_data = cfg.get('localcontext', 'test_label_data')
output_file = cfg.get('globalcontext', 'test_local_data')
input_data = getting_data(training_input_data, "Input")
label_data = getting_data(training_label_data, "Label")
output_data = getting_data(training_output_data, "Output")
model_local = local_context_learning(input_l, input_d, output_d, hidden_d, filters_n, kernel_v, learning_r, drop_r)
print("Training Local Context Model...")
model_local.fit([input_data, label_data], output_data, batch_size=batch_size_num, epochs=epoch_num)
model_local.save("local_context.h5")
print("Done")
input_data = getting_data(testing_input_data, "Input")
label_data = getting_data(testing_label_data, "Label")
print("Predicting Results for Testing...")
local_context_result = model_local.predict([input_data, label_data])
final_results = np.array(local_context_result)
np.save(output_file, final_results)
print("Done")
