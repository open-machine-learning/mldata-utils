import h5py, numpy
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
from .basehandler import BaseHandler


class H5_RData(BaseHandler):
    """Handle RData files."""

    def __init__(self, *args, **kwargs):
        super(H5_RData, self).__init__(*args, **kwargs)


    def read(self):
        raise Exception("Not supported...")


    def write(self, data):
        group=self.get_data_group(data)
        dest=robjects.globalenv
        if group == 'data':
            datavals = data['data']
            ordering = data['ordering']
            attrlist = []
            nameind = 0
            names = data['names']
            types = data['types']
            for cur_feat in ordering:
                if len(datavals[cur_feat].shape) > 1:
                    for k in range(datavals[cur_feat].shape[0]):
                        if str(types[nameind]).startswith('nominal'):
                            attrlist.append((names[nameind], robjects.FactorVector(robjects.StrVector(datavals[cur_feat][k]))))
                        else:
                            attrlist.append((names[nameind], datavals[cur_feat][k]))
                        nameind += 1
                else:
                    if str(types[nameind]).startswith('nominal'):
                        attrlist.append((names[nameind], robjects.FactorVector(robjects.StrVector(datavals[cur_feat]))))
                    else:
                        attrlist.append((names[nameind], datavals[cur_feat]))
                    nameind += 1
            dest[data['name']] = robjects.DataFrame(rlc.OrdDict(attrlist))
        elif group == 'task':
            d=data[group]
            for k in list(d.keys()):
                dest[k] = d[k]
        robjects.r.save(*list(robjects.r.ls(dest)), file=self.fname)
