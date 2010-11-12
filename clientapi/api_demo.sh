#!/bin/bash
# A small demo to show how to use the command line interface to mldata.org



# Download the dataset and tasks
curl http://mldata.org/repository/data/download/datasets-uci-iris/ > iris.h5
curl http://mldata.org/repository/task/download/iris_classification/ > iris_classification.h5
curl http://mldata.org/repository/task/download/iris_regression/ > iris_regression.h5
curl http://mldata.org/repository/data/download/datasets-uci-diabetes/ > pima.h5
curl http://mldata.org/repository/task/download/diabetes-classification/ > pima_task.h5

# do some machine learning
python mlprocess.py iris_classification.h5 iris.h5 iris_preds.txt 
python mlprocess.py iris_regression.h5 iris.h5 iris_regs.txt
python mlprocess.py pima_task.h5 pima.h5 pima_preds.txt


