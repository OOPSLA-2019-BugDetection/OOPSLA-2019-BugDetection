# Improving Bug Detection via Context-based Code Representation Learning and Attention-based Neural Networks

Bug detection has been shown to be an effective way to help developers in detecting bugs early, thus, saving
much effort and time in software development process. Recently, deep learning-based bug detection approaches
have gained successes over the traditional machine learning-based approaches, the rule-based program analysis
approaches, and mining-based approaches. However, they are still limited in detecting bugs that involve
multiple methods and suffer high rate of false positives. In this paper, we propose a combination approach of
the use of contexts and attention neural network to overcome those limitations. We propose to use as the
global context the Program Dependence Graph (PDG) and Data Flow Graph (DFG) to connect the method
under investigation with the other relevant methods that might contribute to the buggy code. The global
context is complemented by the local context extracted from the path on the AST built from the method’s
body. The use of PDG and DFG enables our model to reduce the false positive rate, while to complement
for the potential reduction in recall, we make use of the attention neural network mechanism to put more
weights on the buggy paths in the source code. That is, the paths that are similar to the buggy paths will be
ranked higher, thus, improving the recall of our model. We have conducted several experiments to evaluate
our approach on a very large dataset with +4.973M methods in 92 different project versions. The results show
that our tool can have a relative improvement up to 160% on F score when comparing with the state-of-the-art
bug detection approaches. Our tool can detect 48 true bugs in top 100 reported ones, which is 24 more true
bugs when comparing with other baselines. We also reported that our representation is better suitable for bug
detection and relatively improves over the other representations up to 206% in accuracy.

----------
# Data Set

The data can be downloaded from https://drive.google.com/open?id=1iD4d1WpKmGVgqADkhhFFrOUS5RM53nbg and https://drive.google.com/open?id=1LjsypaXbbmhIB05LYdR-sGmnmqZQdvOy

The data is also available at https://zenodo.org/record/3719225 and https://zenodo.org/record/3719219 (Thanks for monperrus's help and effort)

Our data set include two files:

1. patch_files.tar.gz: It contains the bug fix code for each bug id. The patch files are separated by the project. All versions are included together in order to avoid that one bug fix can influence more than one version of code. If you want to download from zenodo, download: `curl -LO "https://zenodo.org/record/3719219/files/patch_files.tar.gz?download=1"`. If you want to download from google drive, you can download directly.

2. detection_data.tar.gz: It contains all code for each version and a bug summary file for each version. The data is separated by the project, and in each project, code is separated by the version with the different folders and a summary file. The data in the summary file which is the pickle file is a 2-dimensional array with shape [N, 6], N is the number of methods with the bug, 6 is the number of info one method carries with format [method name, class name or 'NaN' if no class, source code, bug id, java file name path, affect version id].

In order to run our model, you need to download these two files. If you want to run our model in a different dataset, you need to format your data set into the same format as our data set. 

## To build your own dataset

If you want to run our model on your own dataset, you can prepare your data in the following format:

1. You need to have two folders to store training data and testing data

2. For both training and testing data folder, you could put sub-folders inside. Each sub-folder is ONE Java project.

3. In order to reduce the influence of unrelated information, you would better remove all comments from the code. (Our model can ignore the comment itself, but you can also do it on the dataset to avoid possible problems.)

4. Inside of the training data folder and testing data folder, you need to have one JSON file in each of them to point out which line in which file is buggy in order to make the model can learn and test based on your data. These two files you need to name them into Buggy_lines.json. For each of them, the structure should like this:

{"File_path_1": [1, 5,..., n], "File_path_2": [3, 6,..., m],...}

The "File_path" is the relative path of your file, for example:

project_1/src/test.java (The absolute path of it could be like "/data/training/project_1/src/test.java")

And the buggy line numbers should be in a list.

----------

# Run our model

## 1. Config.ini:

Set the all required settings in config.ini. All required parts marked as (FILL). And detailed explanation for each option is in config.ini.

## 2. Run Model

Run "main.py" to get the evaluation results and running time.

If you have any question about the code, please directly email me @ yl622@njit.edu
