from distutils.core import setup

VERSION='0.4.0'
DESCRIPTION='Utilities around hdf5 and for compute performance measures',
LONG_DESCRIPTION="""
Utility functions to perform evaluations and to convert from/to the hdf5 based mldata format into/from typical machine learning formats like .csv,.arff,.octave,.mat. This code is actually used on mldata.org to perform the evaluations and data conversions.
"""
AUTHOR = ['Sebastian Henschel', 'Mikio Braun', 'Hagen Zahn','Cheng Soon Ong', 'Soeren Sonnenburg']
AUTHOR_EMAIL = ['mldata@kodeaffe.de', 'mikio@cs.tu-berlin.de', 'hzahn@informatik.hu-berlin.de',
                'chengsoon.ong@inf.ethz.ch', 'Soeren.Sonnenburg@tu-berlin.de']
URL = 'http://mloss.org/software/view/262/'

LICENSE = """
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, check
http://www.gnu.org/licenses/gpl-3.0.html
"""

setup(
    name='mldata-utils',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    packages=['ml2h5', 'ml2h5.converter', 'mleval'],
    scripts=['scripts/ml2h5conv', 'scripts/ml2h5extract_data', 'scripts/ml2h5extract_task'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Utilities',
    ],
    platforms=['POSIX'],
)
