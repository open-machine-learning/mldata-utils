"""A class to deal with standard prediction outputs.
Currently supports:
- binary classification
- multiclass
- regression
"""

from numpy import array, sign, unique


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
        all_names = list(map(str, all_names))
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
        all_names = list(map(str, all_names))
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
        return list(map(str, preds))

def init_output(task_type):
    """A factory for creating the right output class"""
    output_types = {'Binary Classification': OutputsBinClass,
                    'Multi Class Classification': OutputsMultiClass,
                    'Regression': OutputsRegression,
                    }
    return output_types[task_type]()

