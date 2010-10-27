import numpy, h5py, os, copy
from scipy.sparse import csc_matrix
from basehandler import BaseHandler



class H5_LibSVM(BaseHandler):
    """Handle LibSVM files."""

    def __init__(self, *args, **kwargs):
        """Constructor.

        @ivar label_maxidx: highest index for a label
        @type label_maxidx: integer
        @ivar is_multilabel: if data is of type multilabel
        @type is_multilabel: boolean
        """
        super(H5_LibSVM, self).__init__(*args, **kwargs)
        self.label_maxidx = 0
        self.is_multilabel = False


    def _scrub_labels(self, label):
        """Convert labels to doubles and determine max index

        @param label: labels read from data file
        @type label: list of characters
        @return: labels converted to double
        @rtype: list of integer
        """
        str_label = ''.join(label)
        if not self.is_multilabel and str_label.find(',') == -1:
            self.label_maxidx = 0
            return [numpy.double(str_label)]
        else:
            self.is_multilabel = True
            lab = str_label.split(',')
            for i in xrange(len(lab)):
                if not lab[i]:
                    lab[i] = 0
                else:
                    # int conversion to prevent error msg
                    lab[i] = int(float((lab[i])))
                if lab[i] > self.label_maxidx:
                    self.label_maxidx = lab[i]
            return lab


    def _parse_line(self, line):
        """Parse a LibSVM input line and return attributes.

        @param line: line to parse
        @type line: string
        @return: variables in this line
        @rtype: list of variables
        """
        state = 'label'
        idx = []
        val = []
        label = []
        variables = []
        for c in line:
            if state == 'label':
                if c.isspace():
                    state = 'idx'
                    label = self._scrub_labels(label)
                else:
                    label.append(c)
            elif state == 'idx':
                if not c.isspace():
                    if c == ':':
                        state = 'preval'
                    else:
                        idx.append(c)
            elif state == 'preval':
                if not c.isspace():
                    val.append(c)
                    state = 'val'
            elif state == 'val':
                if c.isspace():
                    variables.append([int(''.join(idx)), ''.join(val)])
                    idx = []
                    val = []
                    state = 'idx'
                else:
                    val.append(c)

        return {'label':label, 'variables':variables}


    def _get_parsed_data(self):
        """Retrieves a SciPy Compressed Sparse Column matrix and labels from file.

        @return: compressed sparse column matrix + labels
        @rtype: list of scipy.sparse.csc_matrix and label tuple/2-d tuple (multilabel)
        """
        parsed = []
        infile = open(self.fname, 'r')
        for line in infile:
            parsed.append(self._parse_line(line))
        infile.close()

        indices_var = []
        indices_lab = []
        indptr_var = [0]
        indptr_lab = [0]
        ptr_var = 0
        ptr_lab = 0
        data_var = []
        data_lab = []
        label = []
        for i in xrange(len(parsed)):
            if self.is_multilabel: # -> values are indices
                for idx in parsed[i]['label']:
                    indices_lab.append(int(idx))
                    data_lab.append(1.)
                    ptr_lab += 1
                indptr_lab.append(ptr_lab)
            else: # only single label -> value is actual value
                label.append(parsed[i]['label'])

            for v in parsed[i]['variables']:
                indices_var.append(int(v[0]) - 1) # -1: (multi)label idx
                data_var.append(numpy.double(v[1]))
                ptr_var += 1
            indptr_var.append(ptr_var)

        if self.is_multilabel:
            label = csc_matrix(
                (numpy.array(data_lab), numpy.array(indices_lab), numpy.array(indptr_lab))
            ).todense()
        else:
            label = numpy.matrix(label)

        import pdb
        pdb.set_trace()

        return (
            csc_matrix(
                (numpy.array(data_var), numpy.array(indices_var), numpy.array(indptr_var))
            ),
            label,
        )


    def get_comment(self):
        return 'LibSVM'


    def read(self):
        (A, label) = self._get_parsed_data()
        data = { 'label' : label }

        if A.nnz/numpy.double(A.shape[0]*A.shape[1]) < 0.5: # sparse
            data['data_indices'] = A.indices
            data['data_indptr'] = A.indptr
            data['data_data'] = A.data
        else: # dense
            data['data'] = A.todense()

        import pdb
        pdb.set_trace()

        return {
            'name': self.get_name(),
            'comment': 'LibSVM',
            'ordering': ['label', 'data'],
            'names': ['label', 'data'],
            'data': data,
        }


    def write(self, data):
        """ this needs to `transpose' the data """
        libsvm = open(self.fname, 'w')

        if 'label' in data.keys():
            import pdb
            pdb.set_trace()
            if len(data['data']['label'][0]) == 1:
                is_multilabel = False
            else:
                is_multilabel = True

        lengths=dict()
        for o in data['ordering']:
            x=numpy.array(data['data'][o])
            if numpy.issubdtype(x.dtype, numpy.int):
                data['data'][o]=x.astype(numpy.float64)
            try:
                lengths[o]=data['data'][o].shape[0]
            except AttributeError:
                lengths[o]=len(data['data'][o])
        l=set(lengths.values())
        import pdb
        pdb.set_trace()
        assert(len(l)==1)
        l=l.pop()

        import pdb
        pdb.set_trace()
        for i in xrange(l):
            out = []
            for o in data['ordering']:
                d=data['data'][o]
                if o == 'label':
                    if is_multilabel:
                        labels = []
                        for j in xrange(len(d[i])):
                            if d[i][j] == 1:
                                labels.append(str(j))
                        out.append(','.join(labels))
                    else:
                        out.append(str(data['label'][i][0]))

            for j in xrange(len(d[i])):
                out.append(str(j+1) + ':' + str(d[i][j]))
            libsvm.write(" ".join(out) + "\n")

        libsvm.close()

        return True
