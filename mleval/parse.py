import numpy

def parse_floats(fil):
    """
    Parse file and return numpy array
    """

    fil.seek(0)

    values=list()
    for l in fil.xreadlines():
        values.append( [ float(e) for e in l.split() ] )

    return numpy.array(values)

def parse_strings(output_file, label_file):
    """
    Parse file expecting a few string objects only converting them to +/-1 and return numpy array
    """

    output_file.seek(0)
    label_file.seek(0)

    output_valid_values=set()
    label_valid_values=set()

    for l in output_file.xreadlines():
        output_valid_values.add(l.strip())

    for l in label_file.xreadlines():
        label_valid_values.add(l.strip())

    if len(output_valid_values)>2 or len(label_valid_values)>2:
        msg="Only Binary outputs are currently supported but found more than two:\n"
        msg+="outputs are: %s\n" % str(output_valid_values)
        msg+="labels are: %s\n" % str(label_valid_values)
        raise Exception(msg)

    if len(output_valid_values-label_valid_values):
        msg="Invalid outputs found\n"
        msg+="outputs are: %s\n" % str(output_valid_values)
        msg+="labels are: %s\n" % str(label_valid_values)
        raise Exception(msg)

    if len(output_valid_values-label_valid_values):
        msg="Invalid outputs found\n"
        msg+="outputs are: %s\n" % str(output_valid_values)
        msg+="labels are: %s\n" % str(label_valid_values)
        raise Exception(msg)

    map_dict=dict()
    i=-1
    for k in label_valid_values:
        map_dict[k]=i
        i+=1

    output_file.seek(0)
    label_file.seek(0)

    output_values=list()
    for l in output_file.xreadlines():
        output_values.append( [ float(map_dict[e.strip()]) for e in l.split() ] )

    label_values=list()
    for l in label_file.xreadlines():
        label_values.append( [ float(map_dict[e.strip()]) for e in l.split() ] )


    return numpy.array(output_values),numpy.array(label_values)


def brute_force_parse(output_file, label_file):
    try:
        outputs=parse_floats(output_file)
        labels=parse_floats(label_file)
        return outputs,labels
    except:
        return parse_strings(output_file, label_file)
