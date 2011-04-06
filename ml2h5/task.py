"""
Handle Task objects and files.

This module heavily relies on the functionality required for http://mldata.org
"""

import os, h5py, numpy
import ml2h5.data
from . import VERSION_MLDATA,NUM_EXTRACT
from indexsplit import reduce_split_str

COMPRESSION = None

task_data_fields = ['train_idx', 'val_idx', 'test_idx', 'data_split', 'input_variables', 'output_variables']
task_descr_fields = ['performance_measure','type']


def update_object(h5, name, value):
    if name in h5.keys():
        del h5[name]

    h5[name] = value

def get_splitnames(fnames):
    """Helper function to get names of splits.

    Get a name like test_idx, train_idx from given filenames.

    @param fnames: filenames to get splitnames from
    @type fnames: list of string
    @return: names of splits
    @rtype: list of strings
    """
    names = []
    for name in fnames:
        n = name.split(os.sep)[-1]
        if n.find('train') != -1 or n.find('.tr') != -1:
            names.append('train_idx')
        elif n.find('.val') != -1:
            names.append('val_idx')
        elif n.find('test') != -1 or n.find('.t') != -1 or n.find('.r') != -1:
            names.append('test_idx')
        else:
            names.append(0)

    # replace unknown name by test if train exists or train if test exists
    if 0 in names:
        if 'train_idx' in names:
            names[names.index(0)] = 'test_idx'
        elif 'test_idx' in names:
            names[names.index(0)] = 'train_idx'

    return names


def get_splitdata(fnames):
    """Get split data.

    @param fnames: filenames of related data files
    @type fnames: list of strings
    """
    names = get_splitnames(fnames)
    data = {}
    offset = 0
    for i in xrange(len(fnames)):
        count = sum(1 for line in open(fnames[i]))
        if names[i] in data: # in case we have multiple train/test idx
            data[names[i]].extend(range(offset, offset+count))
        else:
            data[names[i]] = range(offset, offset+count)
        offset += count

    return data


def add_data(fname, splitnames=None, variables=None):
    """Update a Task file by given splitfiles and input/output variables

    @param fname: name of the Task file
    @type fname: string
    @param splitnames: names of files to contain split data
    @type splitnames: list of strings
    @param variables: 'input': list with indices of the variables to be used to predict the target(s), 'output': index of the variable to be predicted from the features
    @type variables: dict of list of integer and integer
    @return: if Task file could be updated
    @rtype: boolean
    """
    try:
        h5 = h5py.File(fname, 'a')
    except:
        return False

    if not 'task' in h5:
        group = h5.create_group('task')
    else:
        group = h5['task']

    if variables:
        if 'input' in variables:
            group.create_dataset('input_variables', data=variables['input'], compression=COMPRESSION)
        if 'output' in variables:
            group.create_dataset('output_variables', data=variables['output'], compression=COMPRESSION)

    if splitnames:
        data = get_splitdata(splitnames)
        for k,v in data.iteritems():
            if v:
                group.create_dataset(k, data=v, compression=COMPRESSION)

    h5.close()
    return True


def _encode(text):
    """Encode given (utf-8) text to something h5py can digest.

    A bit annoying that this kind of stuff is necessary.

    @param text: text
    @type text: string
    @return: encoded string
    @rtype: string
    """
    try:
        return text.encode('ascii', 'ignore')
    except AttributeError:
        return str(text)


def update_description(h5, task):
    """Update description group in Task file.

    @param h5: opened HDF5 file
    @type h5: h5py.File
    @param task: Task object as of mldata.org
    @type task: repository.Task
    @return: if update was successful
    @rtype: boolean
    """

    if not 'task_descr' in h5:
        group = h5.create_group('task_descr')
    else:
        group = h5['task_descr']

    update_object(group, 'pub_date', _encode(task.pub_date))
    update_object(group, 'version', task.version)
    update_object(group, 'slug', _encode(task.slug.text))
    update_object(group, 'summary', _encode(task.summary))
    update_object(group, 'description', _encode(task.description))
    update_object(group, 'urls', _encode(task.urls))
    # Without saving task, it does not have a primary key
    #  and the following line complains
    task.save()
    update_object(group, 'publications',\
        ''.join([_encode(p.title) for p in task.publications.all()]))

    update_object(group, 'input', _encode(task.input))
    update_object(group, 'output', _encode(task.output))
    update_object(group, 'performance_measure', _encode(task.performance_measure))
    update_object(group, 'type', _encode(task.type))
    update_object(group, 'data', _encode(task.data.name))
    if task.data_heldback:
        update_object(group, 'data_heldback', _encode(task.data_heldback.name))
    update_object(group, 'license', _encode(task.license.name))
    update_object(group, 'tags', _encode(task.tags))

    return True



def update_data(h5, taskinfo=None):
    """Update data group in Task file.

    @param h5: opened HDF5 file
    @type h5: h5py.File
    @param taskinfo: data to write to Task file
    @type taskinfo: dict with indices train_idx, test_idx, input_variables, output_variables
    @return: if update was successful
    @rtype: boolean
    """
    if not 'task' in h5:
        group = h5.create_group('task')
    else:
        group = h5['task']

    if not taskinfo:
        return True
    for name in taskinfo:
        if taskinfo[name] is not None:
            if name in group: del group[name]
            group.create_dataset(name, data=taskinfo[name])

    return True

def check_taskfile(fname):
    try:
        format = ml2h5.fileformat.get(fname)
        if not format in ('matlab','h5','octave'):
            return False
   
        c = ml2h5.converter.Converter(fname,
                '/tmp/dummy_does_not_exist.h5', format_in=format, format_out='h5', attribute_names_first=False, merge=False, type='data')
    except:
        return False
    return True

def get_taskinfo(fname):
    taskinfo = None
    data_size = 0
    format = ml2h5.fileformat.get(fname)
    if not format in ('matlab','h5','octave'):
        raise ml2h5.converter.ConversionError, 'Format not supported (only matlab, \
                                                octave, h5) are supported'
    try:
        c = ml2h5.converter.Converter(fname,
                '/tmp/dummy_does_not_exist.h5', format_in=format, format_out='h5', attribute_names_first=False, merge=False, type='data')
        data = c.read()
        
        for g in ('data','task'):
            for f in task_data_fields:
                if data.has_key(g) and data[g].has_key(f):
                    if not taskinfo:
                        taskinfo=dict()
                    taskinfo[f]=data[g][f]
                    if f in ['train_idx','val_idx','test_idx']:
                        data_size+=len(taskinfo[f])
                    
        if 'data_split' in taskinfo.keys():
            data_size=len(taskinfo['data_split'])
            idx=conv_image2idx(taskinfo['data_split'])
            del taskinfo['datasplit']

            for key in ['train_idx','val_idx','test_idx']:        
                taskinfo[key]=idx[key]
            
        for key in ['train_idx','val_idx','test_idx']:    
            if not key in taskinfo.keys():
                taskinfo[key]=[]        
            if len(taskinfo[key])>0 and type(taskinfo[key][0])!=list:
                taskinfo[key]=[taskinfo[key]]

        taskinfo['data_size']=data_size

    except ml2h5.converter.ConversionError:
        pass

    return taskinfo

def update_or_create(fname, task, taskinfo=None):
    """Update or create Task file with data from given Task object.

    @param fname: full path of Task filename
    @type fname: string
    @param task: Task object as of mldata.org
    @type task: repository.Task
    @param taskinfo: data to write to Task file
    @type taskinfo: dict with indices train_idx, test_idx, input_variables, output_variables
    @return: if file could be updated / created
    @rtype: boolean
    """
    try:
        h5 = h5py.File(fname, 'a')
    except:
        return False


    h5.attrs['name'] = _encode(task.name)
    h5.attrs['mldata'] = VERSION_MLDATA
    h5.attrs['comment'] = 'Task file'

# convert train_idx and test_idx to data_split
    data_size=taskinfo['data_size']
    taskinfo['data_split']=conv_idx2image(taskinfo['train_idx'],taskinfo['val_idx'],taskinfo['test_idx'],data_size)
    del taskinfo['train_idx']
    del taskinfo['val_idx']
    del taskinfo['test_idx']
    error = False
    if not update_description(h5, task):
        error = True
    if not update_data(h5, taskinfo):
        error = True

    h5.close()
    return not error


def get_extract(fname):
    """Get extract of Task file.

    @param fname: name of Task file
    @type fname: string
    @return: datasets from Task file
    @rtype: dict of lists
    """
    extract = {}
    if not h5py.is_hdf5(fname):
        return extract

    h5 = h5py.File(fname, 'r')

    for t in task_data_fields:
        try:
            if len(h5['task'][t][...].shape) > 1:
                extract[t]=[]   
                for l in h5['task'][t][...]:
                    extract[t].append(reduce_split_str(l))
            else:    
                extract[t]=reduce_split_str(h5['task'][t][...])
        except KeyError:
            pass
    for t in task_descr_fields:
        try:
            extract[t]=h5['task_descr'][t][...]
        except KeyError:
            pass
    #import pdb
    #pdb.set_trace()
    max_split_size=10
    split_overflow=False
    split_string_overflow=False
    idx={}
    extract_reduce={}
    if 'train_idx' in h5['task'].keys():
        idx['train_idx']=h5['task/train_idx'][...]    
        for key in ['val_idx','test_idx']:
            if key in h5['task'].keys():
                idx[key]=h5['task'][key][...]
            else:
                idx[key]=[]
               
                
            
    elif 'data_split' in h5['task'].keys():
        idx=conv_image2idx(h5['task/data_split'][...])
    try:
        for key in ['train_idx','val_idx','test_idx']:        
            extract[key]=[reduce_split_str(i) for i in idx[key]]
            extract_reduce[key]=[reduce_split_str(i) for i in idx[key][:NUM_EXTRACT]]

        if len(idx['train_idx']) > NUM_EXTRACT:
            split_overflow=True    

        for key in ['train_idx','val_idx','test_idx']:
            if split_overflow:
                extract_reduce[key].append(['...'])            
            for i in range(len(extract_reduce[key])):
                if len(extract_reduce[key][i]) > max_split_size:
                    extract_reduce[key][i] = extract_reduce[key][i][:max_split_size]
                    extract_reduce[key][i][max_split_size-1] = '...'
                    split_string_overflow=True
    
        num_split_reduce=len(extract_reduce['train_idx'])
        reduce_split_nr=[str(i) for i in range(num_split_reduce)]
        if split_overflow:
            reduce_split_nr[-1]='...'    
        extract['split_idx']=zip(range(len(extract['train_idx'])),extract['train_idx'],extract['val_idx'],extract['test_idx'])
        extract['reduce_split_idx']=zip(reduce_split_nr,extract_reduce['train_idx'],extract_reduce['val_idx'],extract_reduce['test_idx'])
        extract['split_overflow']=split_overflow
        extract['split_string_overflow']=split_string_overflow
    except KeyError:
        extract['split']=[(0,[],[])]   
        extract['split_overflow']=False
    h5.close()
    return extract

def get_split_image(fname,split_nr,norm=1000):
    extract = {}

    if not h5py.is_hdf5(fname):
        return extract
    h5 = h5py.File(fname, 'r')

    path = '/task/data_split' 
    if path in h5:
        try:    
            image_data = h5[path][...][split_nr]
        except IndexError:
            return None    
    else: 
        return None    
    h5.close()
    # normalize to length of norm
    image_norm=numpy.zeros([norm,1])
    for i in xrange(len(image_norm)):
        image_norm[i]=image_data[int(i*(float(len(image_data))/norm))]
    return image_norm.T[0]


def conv_idx2image(train_idx,val_idx,test_idx,last_idx):    
    """Convert all idx to image.

    @param train_idx: Train Indices
    @type train_idx: list of lists of int
    @return: datasplit image
    @rtype: list of int
    """
#    import pdb
#    pdb.set_trace()
    if train_idx==None:
        return None

    dim=len(train_idx)
    image_data=numpy.zeros([dim,last_idx], dtype=numpy.uint8)
    for split_nr in range(dim):
            
        train_split=numpy.array(train_idx[split_nr],dtype=int)   
        try:    
            test_split=numpy.array(test_idx[split_nr],dtype=int)   
        except:
            test_split=numpy.array([],dtype=int) 
        try:
            val_split=numpy.array(val_idx[split_nr],dtype=int)   
        except:
            val_split=numpy.array([],dtype=int)        
    
        try:
            image_data[split_nr][train_split]=1
            image_data[split_nr][val_split]=2
            image_data[split_nr][test_split]=3
        except IndexError:            
            raise ml2h5.converter.ConversionError, 'Index out of Range' 
    return image_data


def conv_image2idx(img):
    """Convert all images to idx.

    @param fname: name of Task file
    @type fname: string
    @return: datasets from Task file
    @rtype: dict of lists
    """
    #import pdb
    #pdb.set_trace()
    train_idx=[]
    val_idx=[]
    test_idx=[]

    for split in img:
        train_idx.append(numpy.array(range(len(split)))[split==1])
        val_idx.append(numpy.array(range(len(split)))[split==2])
        test_idx.append(numpy.array(range(len(split)))[split==3])
    return {'train_idx':train_idx,'val_idx':val_idx, 'test_idx':test_idx }  


def get_variables(fname):
    """Get input/output variables from given Data file.

    @param fname: name of Data file to retrieve variables from
    @type fname: string
    @return: input and output variables
    @rtype: dict of lists 'input' and 'output'
    """
    # FIXME: this might be completely wrong
    bucket, num_attr = ml2h5.data.get_num_instattr(fname)
    return {
        'input': range(num_attr)[1:],
        'output': 0,
    }


def get_test_output(fname):
    """Get test_idx and output_variables from given Task file."""
    if not h5py.is_hdf5(fname):
        return None,None
    h5 = h5py.File(fname, 'r')
    test_idx = image2idx(h5['/task/data_split'][...])['test_idx']
    #test_idx = h5['/task/test_idx'][:]
    output_variables = h5['/task/output_variables'][...]
    h5.close()
    return test_idx, output_variables
