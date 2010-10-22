import h5py, numpy
from basehandler import BaseHandler
from scipy.sparse import csc_matrix

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
	if not meta.has_key('name'):
	    return None	
	if not meta.has_key('dtype'):
	    return None	

	if meta['dtype']=='scalar':
	    data=self._read_matrix(octf,1,1)
	elif meta['dtype']=='matrix':
	    data=self._read_matrix(octf,meta['columns'],meta['rows'])
	elif meta['dtype']=='sparse matrix':
	    data=self._read_sparse_matrix(octf,meta['columns'],meta['rows'])
	elif meta['dtype']=='cell':
	    data=self._read_cellarray(octf,meta['columns'],meta['rows'])
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
		if meta.has_key('name'):
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
		(name,dtype,len,col)=self._read_meta(octf)
		if not dtype=='sq_string':
		    return None
	    	else:
		    data.append(self._read_sq_string(octf,len,col))
	return	data    

    def _read_sparse_matrix(self,octf,col,row):
        """Returns the next sparse matrix in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: csc_matrix 
        """


	tmp_data=numpy.array(self._read_matrix(octf,col,row)).T
	data=csc_matrix((tmp_data[2],(tmp_data[0]-1,tmp_data[1]-1)),shape=(col,row))
	return data

    def _read_matrix(self,octf,col,row):
        """Returns the data of a matrix atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return data: matrix
        """

	data=[]
	line=octf.readline()
	lpos=octf.tell()
        while line and not line.startswith('#'):
            if line.startswith(' '):
                line=line[1:]
            sp=line[:-1].split(' ')
            conv_sp=[]
            try:
                for i in sp:
                    conv_sp.append(int(i))
            except ValueError:
                conv_sp=[]
                try:
                    for i in sp:
                        conv_sp.append(float(i))
                except ValueError:
            	    conv_sp=[]
                    conv_sp.append(sp)

            data.append(conv_sp)
	    lpos=octf.tell()
            line = octf.readline()
	octf.seek(lpos)
	return numpy.array(data)

    def read(self):
        data={}
        names=[]
        octf = open(self.fname, 'r')

        # header check
        if not self._check_header(octf):
            raise ConversionError('Header check failed')

        attr=self._next_attr(octf)
        while attr and attr['name']!='':
            if attr['name']!='__nargin__':
                data[attr['name']]=attr['data']
                names.append(attr['name'])
            attr=self._next_attr(octf)

        if (data.keys==[]):
            raise ConversionError('empty conversion')

        return {
            'name': self.get_name(),
            'comment': 'octave',
            'names':[],
            'ordering':names,
            'data':data,
        }



    def _oct_header(self):
        return '# Created by mldata.org for Octave 3.0.1\n'

    def _print_meta(self,attr,name):
        """Return a string of metainformation

        @return meta: string of attr informations
        """
        meta='# name: ' + str(name) + '\n'
        if type(attr) == numpy.ndarray:
            if attr.shape ==(1,1) or attr.shape==(1,):
                meta+='# type: scalar\n'
            else:
        	meta+='# type: matrix\n'
        	meta+='# rows: ' + str(attr.shape[0]) + '\n'
		try:
		    meta+='# columns: ' + str(attr.shape[1]) + '\n'
		except IndexError:
		    meta+='# columns: 1\n'

	elif type(attr)== csc_matrix:
	    meta+='# type: sparse matrix\n'
	    meta+='# nnz: '+str(attr.nnz) + '\n'
	    meta+='# rows: '+str(attr.shape[0]) + '\n'
	    meta+='# columns: '+str(attr.shape[1]) + '\n'
	elif type(attr) == list:
	    meta+='# type: cell\n'
	    meta+='# rows: 1\n'
	    meta+='# columns: ' + str(len(attr)) + '\n'
	else:
	    meta+='# type: sq_string\n'
	    meta+='# elements: 1\n'
        return meta

    def _print_data(self, attr):
        """Return a string of data  

        @return data: string of attr content
        """
	if attr==None:
	   return ''	
        data=''
	# matrix or scalar
        if type(attr) == numpy.ndarray:
	# scalar	
	    if attr.shape==(1,1):
		data=str(attr[0][0]) + '\n'
	    elif attr.shape==(1,):
		data=str(attr[0]) + '\n'   
	# matrix
	    else:		    
		for i in attr:
		    for j in i:
			if j==int(j):
			    j=int(j)	
			data+=' ' + str(j)
		    data+='\n'
	# sparse matrix
	elif type(attr) == csc_matrix:
	    count=0
	    indices=attr.nonzero()	
	    for i in range(len(attr.data)):
		data+=str(indices[0][i]+1) + ' '  + str(indices[1][i]+1) + ' ' + str(attr.data[i])
		data+='\n'
	# cell array
	elif type(attr) == list:
	    for i in attr:
		data+='# name: <cell-element>\n'
		data+='# type: sq_string\n'
		data+='# elements: 1\n'
		data+='# length: ' + str(len(i)) + '\n'
		data+=str(i) + '\n\n'
	# single string
	else:
	    data= '# length: ' + str(len(attr)) + '\n'	
	    data+=str(attr) + '\n\n'
	return data



    def write(self, data):
        of = open(self.fname,'w')
        out = self._oct_header()

        for o in data['ordering']:
            out += self._print_meta(data['data'][o], o)
            out += self._print_data(data['data'][o])

        of.writelines(out)
        of.close()
