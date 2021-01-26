from configparser import ConfigParser
import os
import subprocess


def get_project(root):
    files = os.listdir(root)
    project = []
    for file in files:
        if os.path.isdir(file):
           project.append(root + "file")
    return project


def get_graph_data(input_path, output_path, label):
    subprocess.call("java -jar " + os.getcwd() + "Graph.jar " + input_path + " " + output_path + "" + label, shell=True)


cfg = ConfigParser()
cfg.read('config.ini')
training_file_root = cfg.get('filepath', 'training')
testing_file_root = cfg.get('filepath', 'testing')
result_file_root = cfg.get('filepath', 'results')
test_projects = get_project(testing_file_root)
train_projects = get_project(training_file_root)
output_test_pdg = result_file_root + "/Graph/testing/pdg"
output_test_dfg = result_file_root + "/Graph/testing/dfg"
output_train_pdg = result_file_root + "/Graph/training/pdg"
output_train_dfg = result_file_root + "/Graph/training/dfg"
print("Generating PDG for training dataset...")
get_graph_data(training_file_root, output_train_pdg, "pdg")
print("Generating DFG for training dataset...")
get_graph_data(training_file_root, output_train_dfg, "dfg")
print("Generating PDG for testing dataset...")
get_graph_data(testing_file_root, output_test_pdg, "pdg")
print("Generating DFG for testing dataset...")
get_graph_data(testing_file_root, output_test_dfg, "dfg")
cfg.set('globalcontext', 'test_dfg_data', output_test_dfg + "/vectors")
cfg.set('globalcontext', 'test_pdg_data', output_test_pdg + "/vectors")
cfg.set('globalcontext', 't_dfg_data', output_train_dfg + "/vectors")
cfg.set('globalcontext', 't_pdg_data', output_train_pdg + "/vectors")
cfg.write(open('config.ini', "w"))


