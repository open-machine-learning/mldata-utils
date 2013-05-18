import h5py, numpy
from .basehandler import BaseHandler
from scipy.sparse import csc_matrix
import ml2h5.converter

class H5_OCTAVE(BaseHandler):
    """Handle Octave files."""

    def __init__(self, *args, **kwargs):
        super(H5_OCTAVE, self).__init__(*args, **kwargs)


    def _check_header(self, octf):
        """evidence of octave conformity (disabled)

        @param octf: octave file
        @type octf: opened File
        @return: if header exists
        @rtype: boolean
        """
        octf.seek(0)
        header=octf.readline(15)
        if header.startswith('# Created by '):
            return True
        else:
            return False

    def _next_attr(self, octf):
        """Returns the next atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return (name,data): name and proper matrix of the atribute
        """
        name=''
        data=[]
        meta = self._read_meta(octf)
        # throw conversion exceptions ?
        if meta==None:
            return None
        if 'name' not in meta:
            return None
        if 'dtype' not in meta:
            return None

        if meta['dtype']=='scalar':
            data=self._read_matrix(octf,1,1)
        elif meta['dtype']=='matrix':
            data=self._read_matrix(octf,meta['columns'],meta['rows'])
        elif meta['dtype']=='int32 matrix':
            data=self._read_matrix(octf,1,1,mtype='int')
        elif meta['dtype']=='sparse matrix':
            data=self._read_sparse_matrix(octf,meta['columns'],meta['rows'])
        elif meta['dtype']=='cell':
            data=self._read_cellarray(octf,meta['columns'],meta['rows'])
        elif meta['dtype']=='string':
            data=self._read_sq_string(octf,meta['length'],meta['elements'])

        elif meta['dtype']=='sq_string':
            data=self._read_sq_string(octf,meta['length'],meta['elements'])
        else:
            # unsupported data type. format exception ?
            return None
        return  {
            'name':meta['name'],
            'data':data,
        }

    def _read_meta(self,octf):
        """Returns the meta information of the next atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return meta: dictionary of attribute meta information
        """
        meta={}
        line = octf.readline()
        if not line:
            return None
        while not line.startswith('#'):
            if line == '':
                return None
            line = octf.readline()
        # metadata
        while line and line.startswith('#'):
            sp=line.split(': ')
            if sp[0]=='# name':
                if 'name' in meta:
                    break
                meta['name']=sp[1][:-1]

            if sp[0]=='# type':
                meta['dtype']=sp[1][:-1]
            if sp[0]=='# rows':
                meta['rows']=int(sp[1])
            if sp[0]=='# columns':
                meta['columns']=int(sp[1])
            if sp[0]=='# elements':
                meta['elements']=int(sp[1])
            if sp[0]=='# length':
                meta['length']=int(sp[1])
            if sp[0]=='# ndims':
                meta['ndims']=int(sp[1])
            lpos=octf.tell()
            line = octf.readline()
        octf.seek(lpos)
        return meta

    def _read_sq_string(self,octf,len,ele):
        """Returns the next string sequence atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: array of strings
        """
        data=''
        data=octf.readline()[:-1]
        return data

    def _read_cellarray(self,octf,col,row):
        """Returns the next cellarray atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: array
        """
        data=[]
        for r in range(row):
            for c in range(col):
                meta=self._read_meta(octf)
                if not meta['dtype']=='sq_string':
                    return None
                else:
                    data.append(self._read_sq_string(octf,meta['length'],meta['elements']))
        return data

    def _read_sparse_matrix(self,octf,col,row):
        """Returns the next sparse matrix in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: csc_matrix
        """

        tmp_data=numpy.array(self._read_matrix(octf,0,0)).T
        data=csc_matrix((tmp_data[2],tmp_data[0:2]-1),shape=(row,col))
        data.sort_indices()

        return data

    def _read_matrix(self,octf,col,row,mtype='float'):
        """Returns the data of a matrix atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: matrix
        """
        data=[]
        line=octf.readline()
        lpos=octf.tell()
        if mtype=='int':
            if line.startswith(' '):
                line=line[1:]
            sp=line[:-1].split(' ')
            row=int(sp[0])
            col=int(sp[1])
            line=octf.readline()

        while line and not line.startswith('#'):
            if line.startswith(' '):
                line=line[1:]
            sp=line[:-1].split(' ')

            if mtype=='int':
                conv_sp=[]
                try:
                    for i in sp:
                        conv_sp.append(int(i))
                except ValueError:
                    raise ConversionError('unexpected data type')    
            if mtype=='float':
                conv_sp=[]
                try:
                    for i in sp:
                        conv_sp.append(float(i))
                except ValueError:
                    raise ConversionError('unexpected data type')    

            data.append(conv_sp)
            lpos=octf.tell()
            line = octf.readline()
        octf.seek(lpos)
        out =  numpy.array(data)
        # unknown matrix shape ?
        if not (row==0 and col==0):
            out.shape=(row,col)
        if out.shape[0]==1:
            out.shape=(out.shape[1],)
        return out

    def read(self):
        data={}
        names=[]
        octf = open(self.fname, 'r')
        # header check
        if not self._check_header(octf):
            raise ml2h5.converter.ConversionError('Header check failed')

        attr=self._next_attr(octf)
        while attr and attr['name']!='':
            if attr['name']!='__nargin__':
                data[attr['name']]=attr['data']
                names.append(attr['name'])
            attr=self._next_attr(octf)

        if (data.keys==[]):
            raise ml2h5.converter.ConversionError('empty conversion')
        return {
            'name': self.get_name(),
            'comment': 'octave',
            'names':[],
            'ordering': names,
            'data': data,
        }


    def _oct_header(self):
        return '# Created by mldata.org for Octave 3.0.1\n'

    def _num_matrix(self,m):
        if type(m)!= numpy.ndarray:
            if type(m)==list:
                try:
                    m=numpy.array(m)
                except:
                    return None
            else:
                return None
        if len(m.shape) == 1:
            out=[]
            for i in m:
                try:
                    if int(i)!=i:
                        raise ValueError
                    out.append(int(i))
                except:
                    try:
                        out.append(float(i))
                    except:
                        return None
        elif len(m.shape) == 2:
            out=[]
            for i in m:
                row=[]
                for j in i:
                    try:
                        if int(j)!=j:
                            raise ValueError
                        row.append(int(j))
                    except:
                        try:
                            row.append(float(j))
                        except:
                            return None
                out.append(row)
        else:
            return None
        return out


    def _print_meta(self,of,attr,name):
        """Return a string of metainformation

        @return meta: string of attr informations
        """
        attr_num=self._num_matrix(attr)
        of.write('# name: ' + str(name) + '\n')
        if type(attr) == numpy.ndarray:
            if attr.shape ==():
                of.write('# type: sq_string\n')
                of.write('# elements: 1\n')
            elif attr.shape ==(1,1) or attr.shape==(1,):
                of.write('# type: scalar\n')
            elif len(attr.shape)==1:
                if attr_num==None:
                    of.write('# type: cell\n')
                    of.write('# rows: 1\n')
                    of.write('# columns: ' + str(len(attr)) + '\n')
                else:    
                    if attr.dtype in ['int32', 'int64']:
                        of.write('# type: int32 matrix\n')   
                        of.write('# ndims 2\n')
                    else:    
                        of.write('# type: matrix\n')
                        of.write('# rows: 1\n')
                        try:
                            of.write('# columns: ' + str(attr.shape[0]) + '\n')
                        except IndexError:
                            of.write('# columns: 1\n')

            else:
                if attr.dtype in ['int32', 'int64']:
                    of.write('# type: int32 matrix\n')   
                    of.write('# ndims 2\n')
                else:   
                    of.write('# type: matrix\n')
                    try:
                        of.write('# rows: ' + str(attr.shape[0]) + '\n')
                    except IndexError:
                        of.write('# rows: 1\n')

                    try:
                        of.write('# columns: ' + str(attr.shape[1]) + '\n')
                    except IndexError:
                        of.write('# columns: 1\n')

        elif type(attr)== csc_matrix:
            of.write('# type: sparse matrix\n')
            of.write('# nnz: '+str(attr.nnz) + '\n')
            of.write('# rows: '+str(attr.shape[0]) + '\n')
            of.write('# columns: '+str(attr.shape[1]) + '\n')
        elif type(attr) == list:
            of.write('# type: cell\n')
            of.write('# rows: 1\n')
            of.write('# columns: ' + str(len(attr)) + '\n')
        else:
            of.write('# type: sq_string\n')
            of.write('# elements: 1\n')
        return True

    def _print_data(self, of, attr):
        """Return a string of data

        @return data: string of attr content
        """
        if attr==None:
            return False
        
        attr_num=self._num_matrix(attr)
        # matrix or scalar or cell array
        if type(attr) == numpy.ndarray:
            # sq_string
            if attr.shape==():
                of.write('# length: ' + str(len(str(attr))) + '\n')
                of.write(str(attr) + '\n\n')
            # scalar
            elif attr.shape==(1,1):
                of.write(str(attr_num[0][0]) + '\n')
            elif attr.shape==(1,):
                of.write(str(attr_num[0]) + '\n')
            # matrix
            elif len(attr.shape)==2:
                # int32 matrix     
                if attr.dtype in ['int32','int64']: 
                    of.write(' ' + str(attr.shape[0]) + ' ' + str(attr.shape[1]) + '\n')    
                    for i in attr:
                        for j in i:
                            of.write(' ' + str(j) + '\n')
                # float matrix            
                else:            
                    for i in attr_num:
                        for j in i:
                            of.write(' ' + str(j))
                        of.write('\n')
            # matrix (vector) or cell array
            elif len(attr.shape)==1:
                # matrix
                if attr_num!=None:
                    # int32 matrix     
                    if attr.dtype in ['int32', 'int64']: 
                        of.write(' 1 ' + str(attr.shape[0]) + '\n')    
                        for i in attr:
                            of.write(' ' + str(i) + '\n')
                    # float matrix            
                    else:            
                        for i in attr_num:
                            of.write(' ' + str(i))
                        of.write('\n')
                else:
                    # cell array
                    for i in attr:
                        of.write('# name: <cell-element>\n')
                        of.write('# type: sq_string\n')
                        of.write('# elements: 1\n')
                        of.write('# length: ' + str(len(i)) + '\n')
                        of.write(str(i) + '\n\n')

        # sparse matrix
        elif type(attr) == csc_matrix:
            dat=self._num_matrix(attr.data)
            indptr=attr.indptr
            indices=attr.indices

            for i in range(attr.shape[1]):
                out=[]
                for j in range(indptr[i],indptr[i+1]):
                    if dat[j]==int:
                        out.append("%d %d %d\n" % (indices[j]+1, i+1, dat[j]))
                    else:
                        out.append("%d %d %.15g\n" % (indices[j]+1, i+1, dat[j]))
                of.write(''.join(out))

            # more clean but slower code
            #indices=attr.nonzero()
            #for i in range(len(attr.data)):
            #    of.write(str(indices[0][i]+1) + ' '  + str(indices[1][i]+1) + ' ' + str(dat[i]) + '\n')
        # cell array
        elif type(attr) == list:
            for i in attr:
                of.write('# name: <cell-element>\n')
                of.write('# type: sq_string\n')
                of.write('# elements: 1\n')
                of.write('# length: ' + str(len(i)) + '\n')
                of.write(str(i) + '\n\n')
        # single string
        else:
            of.write('# length: ' + str(len(attr)) + '\n')
            of.write(str(attr) + '\n\n')
        return True


    def write(self, data):
        group=self.get_data_group(data)
        of = open(self.fname,'w')
        of.writelines(self._oct_header())
        for o in data['ordering']:
            self._print_meta(of,data[group][o], o)
            self._print_data(of,data[group][o])

        of.close()
