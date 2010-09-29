import numpy, h5py, os, copy, numpy
from scipy.sparse import csc_matrix
from basehandler import BaseHandler

COMMENT = '# '
SEPERATOR = ','

class H5_CSV(BaseHandler):
    """Handle CSV files.

    @ivar attribute_names_first: first line contains attributes
    @type attribute_names_first: boolean
    """

    def __init__(self, *args, **kwargs):
        super(H5_CSV, self).__init__(*args, **kwargs)
        self.attribute_names_first = False
        self.seperator = SEPERATOR # maybe more flexible in the future


    def _find_nan(self, value):
        """Find NaN values and make them proper numpy.nan.

        Used by map function in get_contents()

        @param value: value to check if it is NaN
        @type value: string
        """
        if value == '?':
            return numpy.nan
        else:
            # str() important! when run thru webserver it will be unicode
            # otherwise and somehow results somewhere in nan being converted
            # to string 'na'...
            return str(value)


    def read(self):
        """
        """
        data = {}
        names = []
        ordering = []

        infile = open(self.fname, 'r')
        parsed = []
        anf = self.attribute_names_first # copy value
        for line in infile:
            l = line.strip().split(self.seperator)
            if anf:
                names = l
                anf = False
            else:
                l = map(self._find_nan, l)
                parsed.append(l)
        infile.close()
        A = numpy.matrix(parsed).T

        for i in xrange(A.shape[0]):
            items = A[i].tolist()[0]
            t = self.get_datatype(items)
            if t == numpy.int32:
                name = 'int' + str(i)
            elif t == numpy.double:
                name = 'double' + str(i)
            else:
                name = 'str' + str(i)
            data[name] = numpy.array(items).astype(t)
            ordering.append(name)

        if not names:
            names = copy.copy(ordering)
        return {
            'name': self.get_name(),
            'comment': 'CSV',
            'names': names,
            'ordering': ordering,
            'data': data,
        }


    def write(self, data):
        csv = open(self.fname, 'w')
        try:
            for i in xrange(len(data['data'])):
                line = map(str, data['data'][i])
                if 'label' in data:
                    label = map(str, data['label'][i])
                    label = self.seperator.join(label)
                    line.insert(0, label)
                csv.write(self.seperator.join(line) + "\n")
        except KeyError, e:
            csv.close()
            os.remove(self.fname)
            raise KeyError(e)
        else:
            csv.close()

        return True
