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
KTF.tensorflow_backend.set_session(session)


def adding_attention(input_tensor, attention_vec, input_length, attention_dim):
    ten_remove = Lambda(lambda x: x[:, :input_length-1])(input_tensor)
    ten_attention = Reshape((1, attention_dim))(attention_vec)
    new_tensor = Concatenate(axis=1)([ten_remove, ten_attention])
    return new_tensor


def local_context_learning(input_length, input_dim, output_dim, hidden_dim, filters_num, kernel_val, learning_rate, drop_rate):
    basic_input = Input(shape=(input_length, input_dim))
    label_input = Input(shape=(1,))
    weighted_input = adding_weight()(basic_input)

    def true_process():
        return weighted_input

    def false_process():
        return basic_input

    actual_input = Lambda(lambda x: tf.cond(x > tf.constant(value=0.5), true_fn=true_process(), false_fn=false_process()))(label_input)
    rnn_output = GRU(hidden_dim, return_sequences=True)(actual_input)
    rnn_att = SeqSelfAttention(attention_activation='sigmoid')(rnn_output)
    cnn_output = Conv1D(filters=filters_num, kernel_size=kernel_val, padding="same")(actual_input)
    cnn_output_reformat = Dense(hidden_dim)(cnn_output)
    cnn_att = SeqSelfAttention(attention_activation='sigmoid')(cnn_output_reformat)
    fixed_rnn_output = adding_attention(rnn_output, rnn_att, input_length, hidden_dim)
    fixed_cnn_output = adding_attention(cnn_output_reformat, cnn_att, input_length, hidden_dim)
    new_value = Concatenate(axis=1)([fixed_rnn_output, fixed_cnn_output])
    new_keys = ones_like(new_value)
    new_result = MultiHeadAttention(head_num=2)([actual_input, new_keys, new_value])
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
input_l = cfg.get('localcontext', 'input_length')
input_d = cfg.get('localcontext', 'input_dim')
output_d = cfg.get('localcontext', 'output_dim')
hidden_d = cfg.get('localcontext', 'hidden_dim')
filters_n = cfg.get('localcontext', 'filters_number')
kernel_v = cfg.get('localcontext', 'kernel_value')
batch_size_num = cfg.get('localcontext', 'batch_size_num')
epoch_num = cfg.get('localcontext', 'epoch_num')
learning_r = cfg.get('localcontext', 'learning_rate')
drop_r = cfg.get('localcontext', 'dropout_rate')
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
local_context_result = model_local.predict([testing_input_data, testing_label_data])
final_results = np.array(local_context_result)
np.save(output_file)
print("Done")
