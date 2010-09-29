"""
Utility functions around HDF5, mostly for mldata.org.
"""

VERSION = '0.3.7'
PROGRAM_NAME = 'ml2h5'
DESCRIPTION = 'Handle machine learning data in HDF5'
LONG_DESCRIPTION = """
Utility functions around HDF5, mostly for mldata.org
"""
AUTHOR = ['Sebastian Henschel', 'Mikio Braun', 'Hagen Zahn','Cheng Soon Ong']
AUTHOR_EMAIL = ['mldata@kodeaffe.de', 'mikio@cs.tu-berlin.de', 'hzahn@informatik.hu-berlin.de',
                'chengsoon.ong@inf.ethz.ch']
URL = 'http://mloss.org/'
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

__all__ = ['task', 'data', 'fileformat', 'converter']
VERSION_MLDATA = '0'
NUM_EXTRACT = 10
COMPRESSION = None
