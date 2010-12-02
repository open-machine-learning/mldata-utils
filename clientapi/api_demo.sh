#!/bin/bash
# A small demo to show how to use the command line interface to mldata.org

# Download the dataset and tasks
curl http://mldata.org/repository/data/download/diabetes_scale/ > pima.h5
curl http://mldata.org/repository/task/download/pima-binclass/ > pima_task.h5
curl http://mldata.org/repository/task/download/pima-binclass-roc/ > pima_aroc.h5
    
curl http://mldata.org/repository/data/download/datasets-uci-iris/ > iris.h5
curl http://mldata.org/repository/task/download/iris-multiclass/ > iris_task.h5
    
curl http://mldata.org/repository/data/download/regression-datasets-housing/ > boston.h5
curl http://mldata.org/repository/data/download/housing_scale/ > boston_scaled.h5
curl http://mldata.org/repository/task/download/boston-housing-regression/ > boston_task.h5
curl http://mldata.org/repository/task/download/boston-housing-scaled-regression/ > boston_task2.h5

# do some machine learning
python mlprocess.py pima_task.h5 pima.h5 pima_preds.txt
python mlprocess.py pima_aroc.h5 pima.h5 pima_preds2.txt # same predictions

python mlprocess.py iris_task.h5 iris.h5 iris_preds.txt

python mlprocess.py boston_task.h5 boston.h5 boston_preds.txt # performs badly due to scaling
python mlprocess.py boston_task2.h5 boston_scaled.h5 boston_preds2.txt
