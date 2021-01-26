#! /bin/bash

python3 Code_bug/localcontext_and_graph.py

for file in ./Code_bug/Graph/testing/pdg
do
    if test -f $file
    then
         python2 node2vec/src/main.py --input Code_bug/Graph/testing/pdg/$file --output Code_bug/Graph/testing/pdg/vector/$file
    fi
for file in ./Code_bug/Graph/testing/pdg
do
    if test -f $file
    then
         python2 node2vec/src/main.py --input Code_bug/Graph/testing/dfg/$file --output Code_bug/Graph/testing/dfg/vector/$file
    fi
for file in ./Code_bug/Graph/testing/pdg
do
    if test -f $file
    then
         python2 node2vec/src/main.py --input Code_bug/Graph/training/pdg/$file --output Code_bug/Graph/training/pdg/vector/$file
    fi
for file in ./Code_bug/Graph/testing/pdg
do
    if test -f $file
    then
         python2 node2vec/src/main.py --input Code_bug/Graph/training/dfg/$file --output Code_bug/Graph/training/dfg/vector/$file
    fi

python3 Code_bug/bug_detection.py
python3 Code_bug/evaluation.py