from generating_longpaths import generating_longpaths as gl
from configparser import ConfigParser
import subprocess
import gensim.models as g
import json
import os
import numpy as np


def get_word2vec_list(file):
    input_file = open(file, "r")
    lines = input_file.readlines()
    lines = lines[1:]
    dic = {}
    for line in lines:
        word = line.strip().split(" ")[0]
        vector = line.strip().split(" ")[1:]
        dic[word] = vector
    return dic


def vector_replace(list_in, dic):
    for i in range(len(list_in)):
        if list_in[i] in dic.keys():
            list_in[i] = dic[list_in[i]]
        else:
            list_in[i] = 0
    return list_in


def get_amount(list_in):
    path_num = 0
    max_token = 0
    for i in range(len(list_in)):
        path_num = path_num + 1
        path_token = 0
        for j in range(len(list_in[i])):
            path_token = path_token + 1
        if path_token > max_token:
            max_token = path_token
    return [path_num, max_token]


def transfor_2_np(list_in, x, y, z):
    if x == 0:
        output = np.zeros([y, z])
        for i in range(len(list_in)):
            for j in range(len(list_in[i])):
                output[i,j] = list_in[i][j]
    else:
        output = np.zeros([x, y, z])
        for i in range(len(list_in)):
            for j in range(len(list_in[i])):
                if type(list_in[i][j]).__name__ == 'list':
                    for k in range(len(list_in[i][j])):
                        output[i][j][k] = list_in[i][j][k]
                else:
                    output[i][j][0] = list_in[i][j]
    return output


def go_through_files(root, output_root):
    buggy_line_file = os.path.join(root, "Buggy_lines.json")
    file_ = open(buggy_line_file, "r")
    buggy_lines = json.load(file_)
    check_list = {}
    long_path_list = []
    buggy_states_list = []
    for project_path, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.java'):
                buggy_line_infile = buggy_lines[os.path.join(project_path, filename).replace(root, "")]
                buggy_line_infile = list(map(int, buggy_line_infile))
                print("java -jar " + os.getcwd() + "Code_bug/ASTgenerate.jar " + os.path.join(project_path, filename) + " " + os.path.join(output_root, "temp.txt"))
                subprocess.call("java -jar " + os.getcwd() + "Code_bug/ASTgenerate.jar " + os.path.join(project_path, filename) + " " + os.path.join(output_root, "temp.txt"), shell=True)
                get_path = gl(os.path.join(output_root, "temp.txt"))
                long_paths = get_path.run()
                for method in long_paths:
                    long_path_set = method[0]
                    line_cover = method[1]
                    for i in range(len(long_path_set)):
                        if filename in check_list.keys():
                            check_list[filename].append([long_path_set[i], line_cover[i]])
                        else:
                            new_list = [[long_path_set[i], line_cover[i]]]
                            check_list[filename] = new_list
                        long_path_list.append(long_path_set[i])
                        if int(line_cover[i][0]) in buggy_line_infile:
                            if len(line_cover[i]) > 1:
                                if int(line_cover[i][1]) in buggy_line_infile:
                                    buggy_states_list.append([1])
                                else:
                                    buggy_states_list.append([0])
                            else:
                                buggy_states_list.append([1])
                        else:
                            buggy_states_list.append([0])
    path_num = get_amount(long_path_list)[0]
    path_token = get_amount(long_path_list)[1]
    return [path_num, path_token, check_list, long_path_list, buggy_states_list]


def prepare_data_files(path_num, path_token, check_list, long_path_list, buggy_states_list, output_root, w_size, w_window, w_workers, label):
    with open(os.path.join(output_root, "check_list.json"), "a") as file_:
        json.dump(check_list, file_)
    word2vec_model = g.Word2Vec(long_path_list, size=w_size, window=w_window, workers=w_workers)
    word2vec_model.wv.save_word2vec_format(os.path.join(output_root, "word2vec.txt"), binary=False)
    diction = get_word2vec_list(os.path.join(output_root, "word2vec.txt"))
    for i in range(len(long_path_list)):
        long_path_list[i] = vector_replace(long_path_list[i], diction)
    long_path_list_np = transfor_2_np(long_path_list, path_num, path_token, w_size)
    np.save(os.path.join(output_root, label + "_input.npy"), long_path_list_np)
    buggy_states_list = transfor_2_np(buggy_states_list, 0, path_num, 1)
    np.save(os.path.join(output_root, label + "_label.npy"), buggy_states_list)
    output_list = buggy_states_list.reshape([path_num, 1])
    output_list = output_list.repeat(w_size, axis=1)
    np.save(os.path.join(output_root, label + "_output.npy"), output_list)


cfg = ConfigParser()
cfg.read('config.ini')
training_file_root = cfg.get('filepath', 'training')
testing_file_root = cfg.get('filepath', 'testing')
result_file_root = cfg.get('filepath', 'results')
word2vec_size = int(cfg.get('localcontext', 'input_dim'))
word2vec_window = int(cfg.get('localcontext', 'word_window'))
word2vec_workers = int(cfg.get('localcontext', 'word_workers'))
output_train = go_through_files(training_file_root, result_file_root)
output_test = go_through_files(testing_file_root, result_file_root)
if output_train[0] > output_test[0]:
    path_n = output_train[0]
else:
    path_n = output_test[0]
if output_train[1] > output_test[1]:
    path_t = output_train[1]
else:
    path_t = output_test[1]
prepare_data_files(path_n, path_t, output_train[2], output_train[3], output_train[4], result_file_root, word2vec_size, word2vec_window, word2vec_workers, "train")
prepare_data_files(path_n, path_t, output_test[2], output_test[3], output_test[4], result_file_root, word2vec_size, word2vec_window, word2vec_workers, "test")
cfg.set('localcontext', 'input_length', str(path_t))
cfg.set('localcontext', 'output_dim', str(word2vec_size))
cfg.set('localcontext', 't_input_data', os.path.join(result_file_root, "train_input.npy"))
cfg.set('localcontext', 't_label_data', os.path.join(result_file_root, "train_label.npy"))
cfg.set('localcontext', 't_output_data', os.path.join(result_file_root, "train_output.npy"))
cfg.set('localcontext', 'test_input_data', os.path.join(result_file_root, "test_input.npy"))
cfg.set('localcontext', 'test_label_data', os.path.join(result_file_root, "test_label.npy"))
cfg.set('globalcontext', 'test_local_data', os.path.join(result_file_root, "test_local_output.npy"))
cfg.write(open('config.ini', "w"))
subprocess.call("python3 " + os.getcwd() + "/Code_bug/model_local.py", shell=True)
