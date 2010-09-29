import h5py, numpy
from basehandler import BaseHandler


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
        return True
        """
        octf.seek(0)
        header=self.readline()
        if header.startswith('# Created by '):
            return True
        else:
        return False
        """

    def _next_attr(self, octf):
        """Returns the next atribute in the octave file

        @param octf: octave file
        @type octf: opened File
        @return (name,data): name and proper matrix of the atribute
        """
        row=1
        col=1
        name='1'
        data=[]

        line = octf.readline()
        while not line.startswith('#'):
            if line == '':
                return {'name':'','data':[]}
            line = octf.readline()


        # metadata
        while line.startswith('#'):
            sp=line.split(': ')
            if sp[0]=='# name':
                name=sp[1][:-1]
            if sp[0]=='# rows':
                rows=int(sp[1])
            if sp[0]=='# columns':
                col=int(sp[1])
            line = octf.readline()

        # matrix
        lpos=octf.tell()
        while (not line.startswith('#')) & (not line == ''):
            if line.startswith(' '):
                line=line[1:]
            sp=line[:-1].split(' ')
            conv_sp=[]
            try:
                for i in sp:
                    conv_sp.append(int(i))
            except ValueError:
                try:
                    for i in sp:
                        conv_sp.append(float(i))
                except ValueError:
                    conv_sp.append(sp)

            data.append(conv_sp)
            lpos=octf.tell()
            line = octf.readline()
            octf.seek(lpos)

        return  {
            'name':name,
            'data':data,
        }



    def read(self):
        data={}
        names=[]
        octf = open(self.fname, 'r')

        # header check
        if not self._check_header(octf):
            raise ConversionError('Header check failed')

        attr=self._next_attr(octf)
        while attr['name']!='':
            if attr['name']!='__nargin__':
                data[attr['name']]=attr['data']
                names.append(attr['name'])
            attr=self._next_attr()

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
        if attr.shape ==(1,):
            meta+='# type: scalar\n'
            return meta
        else: 
            meta+='# type: matrix\n'
        try:
            meta+='# rows: ' + str(attr.shape[1]) + '\n'
        except IndexError:
            meta+='# rows: 1\n'

        try:
            meta+='# columns: ' + str(attr.shape[0]) + '\n'
        except IndexError:
            meta+='# columns: 1\n'

        return meta

    def _print_data(self, attr):
        """Return a string of data  

        @return data: string of attr content
        """
        data=''

        # Vector
        if len(attr.shape) == 1:
            for i in attr:
                data+=' ' +str(i)
            data+='\n'
        # Matrix
        else:
            for i in attr:
                for j in i:
                    data+=' ' + str(j)
                data+='\n'
        return data


    def write(self, data):
        of = open(self.fname,'w')
        out = self._oct_header()

        for i in xrange(len(data['ordering'])):
            out += self._print_meta(data['data'][i], i)
            out += self._print_data(data['data'][i])

        of.writelines(out)
        of.close()
