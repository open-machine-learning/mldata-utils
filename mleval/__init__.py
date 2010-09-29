"""
Utility functions to perform evaluations, mostly for mldata.org.
"""

VERSION = '0.0.1'
PROGRAM_NAME = 'mleval'
DESCRIPTION = 'Compute evaluation scores (accuracy, auROC,...)'
LONG_DESCRIPTION = """
Utility functions to perform evaluations, mostly for mldata.org.
"""
AUTHOR = ['Soeren Sonnenburg']
AUTHOR_EMAIL = ['Soeren.Sonnenburg@tu-berlin.de']
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

__all__ = ['evaluation', 'parse']
