#!/bin/bash
# A small demo to show how to use the command line interface to mldata.org



# Download the dataset and tasks
curl http://mldata.org/repository/data/download/diabetes_scale/ > pima.h5
curl http://mldata.org/repository/task/download/pima-binclass/ > pima_task.h5

# do some machine learning
python mlprocess.py pima_task.h5 pima.h5 pima_preds.txt


