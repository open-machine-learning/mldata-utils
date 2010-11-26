"""
Handle Task objects and files.

This module heavily relies on the functionality required for http://mldata.org
"""

import os, h5py, numpy
import ml2h5.data
from . import VERSION_MLDATA
from indexsplit import reduce_split_str

COMPRESSION = None

task_data_fields = ['train_idx', 'test_idx', 'input_variables', 'output_variables']
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
            names.append('validation_idx')
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

def get_taskinfo(fname):
    taskinfo = None
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
        if not os.path.exists(fname):
            h5 = h5py.File(fname, 'w')
        else:
            h5 = h5py.File(fname, 'a')
    except:
        return False


    h5.attrs['name'] = _encode(task.name)
    h5.attrs['mldata'] = VERSION_MLDATA
    h5.attrs['comment'] = 'Task file'

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
    try:
        h5 = h5py.File(fname, 'r')
    except:
        return extract

    for t in task_data_fields:
        try:
            extract[t]=reduce_split_str(h5['task'][t][...])
        except KeyError:
            pass
    for t in task_descr_fields:
        try:
            extract[t]=h5['task_descr'][t][...]
        except KeyError:
            pass

    h5.close()

    return extract

def get_split_image(fname):
    extract = {}
    try:
        h5 = h5py.File(fname, 'r')
    except:
        return extract

    dsets = ['train_idx', 'test_idx']
    for dset in dsets:
        path = '/task/' + dset
        if path in h5:
            extract[dset] = h5[path][...]
    
    h5.close()

    if len(extract['train_idx'].shape)==1:
        dim=1
    else:
        dim=extract['train_idx'].shape[0]
    image_data=numpy.zeros([max(extract['train_idx'][-1],extract['test_idx'][-1])+1,dim])
    image_data[extract['train_idx']]=1
    image_data[extract['test_idx']]=2
    return image_data.T

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
    h5 = h5py.File(fname, 'r')
    test_idx = h5['/task/test_idx'][:]
    output_variables = h5['/task/output_variables'][...]
    h5.close()
    return test_idx, output_variables
