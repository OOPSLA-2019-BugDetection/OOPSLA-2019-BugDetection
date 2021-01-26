import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling1D, Dropout
from keras.models import Input, Model
from keras.optimizers import adam
from configparser import ConfigParser
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)


def get_method_max_length(method_list):
    max_length = 0
    length_record = 0
    method_id = -1
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            if length_record > max_length:
                max_length = length_record
            length_record = 1
        else:
            length_record = length_record + 1
            if i != len(method_list) - 1 and length_record > max_length:
                max_length = length_record
    return max_length


def get_method_amount(method_list):
    method_num = 0
    method_id = -1
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            method_num = method_num + 1
    return method_num


def grouping_for_method(local_v, dfg_v, pdg_v, method_list, input_length, input_dim):
    combine_1 = local_v*dfg_v
    combine_2 = local_v*pdg_v
    combine = (combine_1 + combine_2)*0.5
    method_num = get_method_amount(method_list)
    method_matrix = np.zeros([method_num, input_length, input_dim])
    method_id = -1
    count = 0
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            count = 0
        for j in range(len(combine[i])):
            method_matrix[method_id][count][j] = combine[i][j]
        count = count + 1
    return method_matrix


def global_context_learning(input_length, input_dim, output_dim, filters_num, kernel_val, learning_rate, drop_rate):
    method_input = Input(shape=(input_length, input_dim))
    cnn_output = Conv2D(filters=filters_num, kernel_size=kernel_val, padding="same")(method_input)
    pooling_output = MaxPooling1D(pool_size=2, padding='same')(cnn_output)
    output = Flatten()(pooling_output)
    output_fix = Dropout(rate=drop_rate)(output)
    fixed_output = Dense(output_dim)(output_fix)
    stand_output = Activation(activation='softmax')(fixed_output)
    model = Model(method_input, stand_output)
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
input_d = cfg.get('globalcontext', 'input_dim')
output_d = cfg.get('globalcontext', 'output_dim')
filters_n = cfg.get('globalcontext', 'filters_number')
kernel_v = cfg.get('globalcontext', 'kernel_value')
batch_size_num = cfg.get('globalcontext', 'batch_size_num')
epoch_num = cfg.get('globalcontext', 'epoch_num')
learning_r = cfg.get('globalcontext', 'learning_rate')
drop_r = cfg.get('globalcontext', 'dropout_rate')
training_local_data = cfg.get('globalcontext', 't_local_data')
training_dfg_data = cfg.get('globalcontext', 't_dfg_data')
training_pdg_data = cfg.get('globalcontext', 't_pdg_data')
training_method_data = cfg.get('globalcontext', 't_method_data')
training_output_data = cfg.get('globalcontext', 't_output_data')
testing_local_data = cfg.get('globalcontext', 'test_local_data')
testing_dfg_data = cfg.get('globalcontext', 'test_dfg_data')
testing_pdg_data = cfg.get('globalcontext', 'test_pdg_data')
testing_method_data = cfg.get('globalcontext', 'test_method_data')
testing_output_data = cfg.get('globalcontext', 'test_output_data')
local_data = getting_data(training_local_data, "Local Context")
dfg_data = getting_data(training_dfg_data, "DFG")
pdg_data = getting_data(training_pdg_data, "PDG")
method_l = getting_data(training_method_data, "Method Belonging")
output_data = getting_data(training_pdg_data, "Output")
input_l = get_method_max_length(method_l)
method_m = grouping_for_method(local_data, dfg_data, pdg_data, method_l, input_l, input_d)
model_local = global_context_learning(input_l, input_d, output_d, filters_n, kernel_v, learning_r, drop_r)
print("Training Global Context Model...")
model_local.fit(method_m, output_data, batch_size=batch_size_num, epochs=epoch_num)
model_local.save("global_context.h5")
print("Done")
local_data = getting_data(testing_local_data, "Local Context")
dfg_data = getting_data(testing_dfg_data, "DFG")
pdg_data = getting_data(testing_pdg_data, "PDG")
method_l = getting_data(testing_method_data, "Method Belonging")
input_l = get_method_max_length(method_l)
method_m = grouping_for_method(local_data, dfg_data, pdg_data, method_l, input_l, input_d)
print("Predicting Results for Testing...")
local_context_result = model_local.predict(method_m)
final_results = np.array(local_context_result)
np.save(testing_output_data)
print("Done")
