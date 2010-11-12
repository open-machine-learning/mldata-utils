"""Interact with mldata.org without clicking on the website"""

from numpy import zeros, array, sign, unique
import h5py
from ml2h5.converter.basehandler import BaseHandler
from shogun.Features import RealFeatures, Labels
from shogun.Kernel import GaussianKernel
from shogun.Classifier import LibSVMMultiClass, LibSVM
from shogun.Regression import LibSVR

class Outputs(object):
    """A container for the labels/predictions of a
    supervised learning method
    """
    def str2label(str_label):
        raise NotImplementedError

    def label2str(preds):
        raise NotImplementedError


class OutputsClassification(Outputs):
    """A container for labels/predictions for classification"""
    name2id = {}
    id2name = {}
    
    def str2label(self, str_label):
        """Convert labels into class ids"""
        self.init_id(str_label)
        int_label = []
        for lab in str_label:
            int_label.append(self.name2id[str(lab[0])])
        return array(int_label, dtype=float)


class OutputsBinClass(OutputsClassification):
    """A container for labels/predictions for
    binary classification
    """
    def init_id(self, str_label):
        """Discover the names of the labels"""
        all_names = unique(str_label)
        assert(len(all_names) == 2)
        self.name2id = {all_names[0]: 1,
                        all_names[1]: -1,
                        }
        self.id2name = {1: all_names[0],
                        -1: all_names[1],
                        }
    
    def label2str(self, preds):
        """Convert predictions into classification labels"""
        pred_label = []
        for pred in preds:
            pred_label.append(self.id2name[sign(pred)])
        return pred_label



class OutputsMultiClass(OutputsClassification):
    """A container for labels/predictions for
    multiclass classification
    """
    def init_id(self, str_label):
        """Discover the names of the labels"""
        all_names = unique(str_label)
        assert(len(all_names) > 2)
        self.name2id = {}
        self.id2name = {}
        for (idx, name) in enumerate(all_names):
            self.name2id[name] = idx
            self.id2name[idx] = name
    
    def label2str(self, preds):
        """Convert predictions into classification labels"""
        pred_label = []
        for pred in preds:
            pred_label.append(self.id2name[pred])
        return pred_label

class OutputsRegression(Outputs):
    """A container for labels/predictions for
    regression
    """
    def str2label(self, str_label):
        """Convert labels into floats"""
        float_label = []
        for lab in str_label:
            float_label.append(float(lab[0]))
        return array(float_label, dtype=float)

    def label2str(self, preds):
        """Convert predictions into strings"""
        return map(str, preds)


def init_output(task_type):
    """A factory for creating the right output class"""
    output_types = {'Binary Classification': OutputsBinClass,
                    'Multi Class Classification': OutputsMultiClass,
                    'Regression': OutputsRegression,
                    }
    return output_types[task_type]()


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

def mlprocess(task_filename, data_filename, pred_filename):
    """Demo of creating machine learning process."""

    df = BaseHandler(data_filename)
    tf = h5py.File(task_filename,'r')

    fidx = tf['task']['input_variables'].value
    lidx = tf['task']['output_variables'].value
    train_idx = tf['task']['train_idx'].value
    test_idx = tf['task']['test_idx'].value
    task_type = tf['task_descr']['type'].value

    all_data = array(df.read()['data'])
    train_data = all_data[train_idx]
    fm_train = train_data[:,fidx].T
    fm_train_real = zeros(fm_train.shape)
    for ridx in range(fm_train.shape[0]):
        for cidx in range(fm_train.shape[1]):
            fm_train_real[ridx,cidx] = float(fm_train[ridx,cidx])

    test_data = all_data[test_idx]
    fm_test = test_data[:,fidx].T
    fm_test_real = zeros(fm_train.shape)
    for ridx in range(fm_test.shape[0]):
        for cidx in range(fm_test.shape[1]):
            fm_test_real[ridx,cidx] = float(fm_test[ridx,cidx])

    outputs = init_output(task_type)
    str_label = train_data[:,lidx]
    label_train = outputs.str2label(str_label)

    feats_train = RealFeatures(fm_train_real)
    feats_test = RealFeatures(fm_test_real)
    width=100.0
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
