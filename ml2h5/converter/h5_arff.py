import h5py, numpy, copy
import arff
from basehandler import BaseHandler


class H5_ARFF(BaseHandler):
    """Handle ARFF files.

    It uses the module arff provided by the dataformat project:
    http://mloss.org/software/view/163/
    """

    def __init__(self, *args, **kwargs):
        super(H5_ARFF, self).__init__(*args, **kwargs)


    def _get_types(self, af):
        """Get types from Arff file

        @param af: opened ARFF file
        @type af: arff.ArffFile
        @return: types in arff file
        @rtype: numpy.array
        """
        # need to get it in the right order
        types = []
        for name in af.attributes:
            t = af.attribute_types[name]
            if af.attribute_data[name]:
                t += ':' + ','.join(af.attribute_data[name])
            types.append(t)
        return numpy.array(types)


    def read(self):
        data = {}
        names = []
        ordering = []
        af = arff.ArffFile.load(self.fname)

        for name in af.attributes:
            names.append(name)
            data[name] = []

        for item in af.data:
            for i in xrange(len(data)):
                data[names[i]].append(item[i])

        # conversion to proper data types
        for name, values in data.iteritems():
            if af.attribute_types[name] == 'date':
                t = self.str_type
            else:
                t = self.get_datatype(values)
            data[name] = numpy.array(values).astype(t)

        return {
            'name': af.relation,
            'comment': af.comment,
            'types': self._get_types(af),
            'names':names,
            'ordering':copy.copy(names),
            'data':data,
        }


    def get_data_as_list(self,data):
        """ this needs to `transpose' the data """

        dl=[]

        lengths=dict()
        for o in data['ordering']:
            try:
                lengths[o]=data['data'][o].shape[0]
            except AttributeError:
                lengths[o]=len(data['data'][o])
        l=set(lengths.values())
        assert(len(l)==1)
        l=l.pop()

        for i in xrange(l):
            line=[]
            for o in data['ordering']:
                try:
                    line.extend(data['data'][o][i])
                except:
                    line.append(data['data'][o][i])
            dl.append(line)
        return dl

    def write(self, data):
        af = arff.ArffFile()
        af.data = self.get_data_as_list(data)
        af.attributes = data['names']
        af.relation = data['name']
        af.comment = data['comment']

        # handle arff types
        if 'types' in data:
            types = data['types']
        else:
            types = []
            for name in af.attributes:
                if name.startswith('int') or name.startswith('double'):
                    types.append('numeric')
                elif name.startswith('date'):
                    types.append('date')
                else:
                    types.append('string')

        for i in xrange(len(types)):
            t = types[i].split(':')
            af.attribute_types[af.attributes[i]] = t[0]
            if len(t) == 1:
                af.attribute_data[af.attributes[i]] = None
            else:
                af.attribute_data[af.attributes[i]] = t[1].split(',')

        af.save(self.fname)
