import multiclass
import classification
import regression
import other

"""
dictionaries of known performance measures:

keys are human readable names, values are tuples of (functioname, description)

pm_hierarchy    - is a dictionary of tasks specific perf measures (e.g. for
                    Classification, Regression, ...).

To add a new measure just add a function, e.g. 

def calcbal(out, lab):
    "Computes the Balanced Error"
    ...

and add it to the the appropriate dictionary

register(pm, 'Balanced Error', calcbal)

"""

pm_hierarchy = { 'Regression': regression.pm,
                 'Binary Classification': classification.pm,
                 'Multi Class Classification' : multiclass.pm,
                 'Multi Output' : other.pm}
