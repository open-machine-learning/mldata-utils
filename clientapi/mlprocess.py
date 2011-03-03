"""Interact with mldata.org without clicking on the website"""

from numpy import zeros
import h5py
from ml2h5.converter.basehandler import BaseHandler
from mleval.output import init_output
from mleval.output import OutputsBinClass, OutputsMultiClass, OutputsRegression
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import GaussianKernel
from shogun.Classifier import LibSVMMultiClass, LibSVM
from shogun.Regression import LibSVR


def init_svm(task_type, kernel, labels):
    """A factory for creating the right svm type"""
    C=1
    epsilon=1e-5
    if task_type == 'Binary Classification':
        svm = LibSVM(C, kernel, labels)
    elif task_type == 'Multi Class Classification':
        svm = LibSVMMultiClass(C, kernel, labels)
    elif task_type == 'Regression':
        tube_epsilon=1e-2
        svm=LibSVR(C, epsilon, kernel, labels)
        svm.set_tube_epsilon(tube_epsilon)
    else:
        print task_type + ' unknown!'

    return svm

def parse_task(task_filename):
    """Manually parse the task file"""

    if not h5py.is_hdf5(task_filename):
        return

    tf = h5py.File(task_filename,'r')

    task_type = tf['task_descr']['type'].value
    fidx = tf['task']['input_variables'].value
    lidx = tf['task']['output_variables'].value
    train_idx = tf['task']['train_idx'].value
    test_idx = tf['task']['test_idx'].value

    return task_type, fidx, lidx, train_idx, test_idx

def parse_data(data_filename):
    """Manually parse the data file"""
    df = BaseHandler(data_filename)
    all_data = df.read_data_as_array()
    return all_data

def _split_data(cur_data, fidx, lidx):
    """Split data into examples and labels"""
    fm = cur_data[:,fidx].T
    ex = zeros(fm.shape)
    for ridx in range(fm.shape[0]):
        for cidx in range(fm.shape[1]):
            ex[ridx,cidx] = float(fm[ridx,cidx])
    lab = cur_data[:,lidx]

    return ex, lab

def split_data(all_data, fidx, lidx, train_idx, test_idx):
    """Split array all_data into training and testing sets,
    as well as examples and labels"""
    train_data = all_data[train_idx]
    train_ex, train_lab = _split_data(train_data, fidx, lidx)
    test_data = all_data[test_idx]
    test_ex, test_lab = _split_data(test_data, fidx, lidx)

    return train_ex, train_lab, test_ex, test_lab

def mlprocess(task_filename, data_filename, pred_filename, verbose=True):
    """Demo of creating machine learning process."""
    task_type, fidx, lidx, train_idx, test_idx = parse_task(task_filename)
    outputs = init_output(task_type)
    all_data = parse_data(data_filename)
    train_ex, train_lab, test_ex, test_lab = split_data(all_data, fidx, lidx, train_idx, test_idx)
    label_train = outputs.str2label(train_lab)

    if verbose:
        print 'Number of features: %d' % train_ex.shape[0]
        print '%d training examples, %d test examples' % (len(train_lab), len(test_lab))

    feats_train = RealFeatures(train_ex)
    feats_test = RealFeatures(test_ex)
    width=1.0
    kernel=GaussianKernel(feats_train, feats_train, width)
    labels=Labels(label_train)
    svm = init_svm(task_type, kernel, labels)
    svm.train()

    kernel.init(feats_train, feats_test)
    preds = svm.classify().get_labels()
    pred_label = outputs.label2str(preds)

    pf = open(pred_filename, 'w')
    for pred in pred_label:
        pf.write(pred+'\n')
    pf.close()




if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print 'Usage: python %s task_file.h5 data_file.h5 preds.txt' %sys.argv[0]
        exit(1)
    task_filename = sys.argv[1]
    data_filename = sys.argv[2]
    pred_filename = sys.argv[3]
    mlprocess(task_filename, data_filename, pred_filename)
