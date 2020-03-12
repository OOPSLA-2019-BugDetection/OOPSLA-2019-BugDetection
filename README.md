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

Our data set include two files:

1. patch_files.tar.gz: It contains the bug fix code for each bug id. The patch files are separated by project. All versions are included together in order to avoid that one bug fix can influence more than one version of code. Link: https://drive.google.com/open?id=1iD4d1WpKmGVgqADkhhFFrOUS5RM53nbg

2. detection_data.tar.gz: It contains all code for each version and a bug summary file for each version. The data is separated by project, and in each project, code is separated by version with different folder and a summary file. The data in the summary file which is the picke file is a 2-dimensional array with shape [N, 6], N is the number of methods with bug, 6 is the number of info one method carries with format [method name, class name or 'NaN' if no class, soure code, bug id, java file name path, affect version id]. Link: https://drive.google.com/open?id=1LjsypaXbbmhIB05LYdR-sGmnmqZQdvOy

In order to run our model, you need to download these two files. If you want to run our model in different data set, you need to format your data set into the same format as our data set. Also, you need to fix the versions.json file. In that file, it stores the information about the projects and versions of these projects.

----------

# Run our model

1. Unzip two data set into two folders.

2. Set the "data_path" in config.ini to the path that you store the data from detection_data.tar.gz. Set the "patch_file_path" in config.ini to the path that you store the data from patch_files.tar.gz. Set the "processed_data_path" to the path that you want to store the processed data for running the model.

3. Run "data_process.py" to process the data and get well formated data set.

4. Run "main.py" to get the evaluation results and running time.
