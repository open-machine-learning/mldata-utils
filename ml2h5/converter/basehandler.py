import os, h5py, numpy
from scipy.sparse import csc_matrix

import ml2h5.task
from ml2h5 import VERSION_MLDATA
from ml2h5.converter import ALLOWED_SEPERATORS


class BaseHandler(object):
    """Base handler class.

    It is the base for classes to handle different data formats.
    It implicitely handles HDF5.

    @cvar str_type: string type to be used for variable length strings in h5py
    @type str_type: numpy.dtype

    @ivar fname: name of file to handle
    @type fname: string
    @ivar seperator: seperator to seperate variables in examples
    @type seperator: string
    """
    str_type = h5py.new_vlen(numpy.str)


    def __init__(self, fname, seperator=None, compression=None, merge=False):
        """
        @param fname: name of in-file
        @type fname: string
        @param seperator: seperator used to seperate examples
        @type seperator: string
        """
        self.fname = fname
        self.compression = compression
        self.set_seperator(seperator)
        self.merge = merge


    def set_seperator(self, seperator):
        """Set the seperator to seperate variables in examples.

        @param seperator: seperator to use
        @type seperator: string
        """
        if seperator in ALLOWED_SEPERATORS:
            self.seperator = seperator
        else:
            raise AttributeError(_("Seperator '%s' not allowed!" % seperator))


    def warn(self, msg):
        """Print a warning message.

        @param msg: message to print
        @type msg: string
        """
        return
        print('WARNING: ' + msg)

    def _convert_to_ndarray(self,path,val):
        """converts a attribut to a set of ndarrays depending on the datatype  

        @param path: path of the attribute in the h5 file
        @type path: string 
        @param val: data of the attribute
        @type val: csc_matrix/ndarray  
        @rtype: list of (string,ndarray) tuples 
        """
        A=val
        out=[]
        dt = h5py.special_dtype(vlen=str)
        if type(A)==csc_matrix: # sparse
            out.append((path+'_indices', A.indices))
            out.append((path+'_indptr', A.indptr))
            out.append((path, A.data))
        elif type(A)==list and len(A)>0 and type(A[0])==str:
            out.append((path, numpy.array(A, dtype=dt)))
        else: # dense
            out.append((path, numpy.array(A)))
        return out

    def get_data_as_list(self,data):
        """ this needs to `transpose' the data """

        dl=[]

        group=self.get_data_group(data)

        lengths=dict()
        for o in data['ordering']:
            x=data[group][o]
            #if numpy.issubdtype(x.dtype, numpy.int):
            #    data[group][o]=x.astype(numpy.float64)
            try:
                lengths[o]=data[group][o].shape[1]
            except (AttributeError, IndexError):
                lengths[o]=len(data[group][o])
        l=set(lengths.values())
        assert(len(l)==1)
        l=l.pop()

        for i in range(l):
            line=[]
            for o in data['ordering']:
                try:
                    line.extend(data[group][o][:,i])
                except:
                    line.append(data[group][o][i])
            dl.append(line)
        return dl

    def get_name(self):
        """Get dataset name from non-HDF5 file

        @return: comment
        @rtype: string
        """
        # without str() it might barf
        return str(os.path.basename(self.fname).split('.')[0])

    def get_data_group(self, data):
        if data and 'group' in data:
            return data['group']
        else:
            return 'data'

    def get_descr_group(self, data):
        if data and 'group' in data:
            return data['group'] + '_descr'
        else:
            return 'data_descr'

    def get_datatype(self, values):
        """Get data type of given values.

        @param values: list of values to check
        @type values: list
        @return: data type to use for conversion
        @rtype: numpy.int32/numpy.double/self.str_type
        """
        dtype = None
        for v in values:
            if isinstance(v, int):
                dtype = numpy.int32
            elif isinstance(v, float):
                dtype = numpy.double
            else: # maybe int/double in string
                try:
                    tmp = int(v)
                    if not dtype: # a previous nan might set it to double
                        dtype = numpy.int32
                except ValueError:
                    try:
                        tmp = float(v)
                        dtype = numpy.double
                    except ValueError:
                        return self.str_type

        return dtype


    def read(self):
        """Get data and description in-memory 

        Retrieve contents from file.

        @return: example names, ordering and the examples
        @rtype: dict of: list of names, list of ordering and dict of examples
        """
        # we want the exception handled elsewhere

        if not h5py.is_hdf5(self.fname):
            return

        h5 = h5py.File(self.fname, 'r')

        contents = {
            'name': h5.attrs['name'],
            'comment': h5.attrs['comment'],
            'mldata': h5.attrs['mldata'],
        }

        if contents['comment']=='Task file':
            contents['task']=dict()
            contents['ordering']=list()
            group='task'
            for field in ml2h5.task.task_data_fields:
                if field in h5[group]:
                    contents['ordering'].append(field)
        else:
            contents['data']=dict()
            contents['ordering']=h5['/data_descr/ordering'][...].tolist()
            group='data'

        contents['group']=group

        if '/%s_descr/names' % group in h5:
           contents['names']=h5['/%s_descr/names' % group][...].tolist()

        if '/%s_descr/types' % group in h5:
            contents['types'] = h5['/%s_descr/types' % group ][...]

        for name in contents['ordering']:
            vname='/%s/%s' % (group, name)
            sp_indices=vname+'_indices'
            sp_indptr=vname+'_indptr'

            if sp_indices in h5['/%s' % group] and sp_indptr in h5['/%s' % group]:
                contents[group][name] = csc_matrix((h5[vname], h5[sp_indices], h5[sp_indptr])
            )
            else:
                d = numpy.array(h5[vname],order='F')

                try:
                    d=d['vlen']
                except:
                    pass
                contents[group][name] = d
        h5.close()
        return contents

    def read_data_as_array(self):
        """Read data from file, and return an array
        
        @return: an array with all data
        @rtype: numpy ndarray
        """
        contents = self.read()
        #group = self.get_data_group(data)
        data = contents['data']
        ordering = contents['ordering']
        if len(data[ordering[0]].shape)>1:
            num_examples = data[ordering[0]].shape[1]
        else:
            num_examples = len(data[ordering[0]])
        data_array = numpy.zeros((0, num_examples))
        for cur_feat in ordering:
            data_array = numpy.vstack([data_array, data[cur_feat]])
        return data_array.T
        

    def _get_merged(self, data):
        """Merge given data where appropriate.

        String arrays are not merged, but all int and all double are merged
        into one matrix.

        @param data: data structure as returned by read()
        @type data: dict
        @return: merged data structure
        @rtype: dict
        """
        merged = {}
        ordering = []
        path = ''
        idx = 0
        merging = None
        group = self.get_data_group(data)

        for name in data['ordering']:
            val = data[group][name]

            if type(val) == csc_matrix:
                merging = None
                path = name
                merged[path] = val
                ordering.append(path)
                continue

            if name.endswith('_indices') or name.endswith('_indptr'):
                merging = None
                path = name
                merged[path] = val
                continue

            if len(val) < 1: continue

            t = type(val[0])
            if t in [numpy.int32, numpy.int64]:
                if merging == 'int':
                    merged[path].append(val)
                else:
                    merging = 'int'
                    path = 'int' + str(idx)
                    ordering.append(path)
                    merged[path] = [val]
                    idx += 1
            elif t == numpy.double:
                if merging == 'double':
                    merged[path].append(val)
                else:
                    merging = 'double'
                    path = 'double' + str(idx)
                    ordering.append(path)
                    merged[path] = [val]
                    idx += 1
            else: # string or matrix
                merging = None
                if name.find('/') != -1: # / sep belongs to hdf5 path
                    path = name.replace('/', '+')
                    data['ordering'][data['ordering'].index(name)] = path
                else:
                    path = name
                ordering.append(path)
                merged[path] = val
        data[group] = {}        
        for k in merged:
            if len(merged[k])==1:
                merged[k] = merged[k][0]    
            data[group][k] = numpy.array(merged[k])
        data['ordering'] = ordering
        return data


    def write(self, data):
        """Write given data to HDF5 file.

        @param data: data to write to HDF5 file.
        @type data: dict of lists
        """
        # we want the exception handled elsewhere
        h5 = h5py.File(self.fname, 'w')
        h5.attrs['name'] = data['name']
        h5.attrs['mldata'] = VERSION_MLDATA
        h5.attrs['comment'] = data['comment']

        data_group = self.get_data_group(data)
        descr_group = self.get_descr_group(data)

        try:
            group = h5.create_group('/%s' % data_group)
            for path, val in data[data_group].items():
                for path, val in self._convert_to_ndarray(path,val):
                    group.create_dataset(path, data=val, compression=self.compression)

            group = h5.create_group('/%s' % descr_group)
            names = numpy.array(data['names']).astype(self.str_type)
            if names.size > 0: # simple 'if names' throws exception if array
                group.create_dataset('names', data=names, compression=self.compression)
            ordering = numpy.array(data['ordering']).astype(self.str_type)
            if ordering.size > 0:
                group.create_dataset('ordering', data=ordering, compression=self.compression)
            if 'types' in data:
                types = numpy.array(data['types']).astype(self.str_type)
                group.create_dataset('types', data=types, compression=self.compression)
        except: # just do some clean-up
            h5.close()
            os.remove(self.fname)
            raise
        else:
            h5.close()
