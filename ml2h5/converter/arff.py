# -*- coding: utf-8 -*-
# Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the names of the Technical University of Berlin, ETH
# Zürich, or Fraunhofer FIRST nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import re, sys, numpy


class ArffFile(object):
    """An ARFF File object describes a data set consisting of a number
    of data points made up of attributes. The whole data set is called
    a 'relation'. Supported attributes are:

        - 'numeric': floating point numbers
        - 'string': strings
        - 'nominal': taking one of a number of possible values

    Not all features of ARFF files are supported yet. The most notable
    exceptions are:

        - no sparse data
        - no support for date and relational attributes

    Also, parsing of strings might still be a bit brittle.

    You can either load or save from files, or write and parse from a
    string.

    You can also construct an empty ARFF file and then fill in your
    data by hand. To define attributes use the define_attribute method.

    Attributes are:

        - 'relation': name of the relation
        - 'attributes': names of the attributes
        - 'attribute_types': types of the attributes
        - 'attribute_data': additional data, for example for nominal attributes.
        - 'comment': the initial comment in the file. Typically contains some
                     information on the data set.
        - 'data': the actual data, by data points.
    """
    def __init__(self):
        """Construct an empty ARFF structure."""
        self.relation = ''
        self.attributes = []
        self.attribute_types = dict()
        self.attribute_data = dict()
        self.comment = []
        self.data = []
        pass

    @staticmethod
    def load(filename):
        """Load an ARFF File from a file."""
        o = open(filename)
        s = o.read()
        a = ArffFile.parse(s)
        o.close()

        if not a.relation:
            raise Exception('Not an ARFF file: ' + filename)

        return a

    def _rm_ticks(self, item):
        """Remove ticks from item if it is a string.

        Some attributes are surrounded by unnecessary ticks.

        @param item: item to check for ticks
        @type item: any, preferably str
        @return: unmodified item or item with ticks removed
        @rtype: type(item)
        """
        try:
            # a few attributes are designated strings by "'"
            if item[0] == "'" and item[-1] == "'":
                return item[1:-1]
        except TypeError:
            pass

        return item

    @staticmethod
    def parse(s):
        """Parse an ARFF File already loaded into a string."""
        a = ArffFile()
        a.state = 'comment'
        a.lineno = 1
        for l in s.splitlines():
            a.__parseline(l)
            a.lineno += 1
        return a

    def save(self, filename):
        """Save an arff structure to a file."""
        o = open(filename, 'w')
        o.write(self.write())
        o.close()

    def write(self):
        """Write an arff structure to a string."""
        o = []
        o.append('% ' + re.sub("\n", "\n% ", self.comment))
        o.append("@relation " + self.esc(self.relation))
        for a in self.attributes:
            at = self.attribute_types[a]
            if at == 'numeric':
                o.append("@attribute " + self.esc(a) + " numeric")
            elif at == 'string':
                o.append("@attribute " + self.esc(a) + " string")
            elif at == 'date':
                o.append("@attribute " + self.esc(a) + " date " + \
                    ''.join(self.attribute_data[a]))
            elif at == 'nominal':
                o.append("@attribute " + self.esc(a) +
                         " {" + ','.join(self.attribute_data[a]) + "}")
            else:
                raise Exception("Type " + at + " not supported for writing!")
        o.append("\n@data")
        for d in self.data:
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at == 'numeric':
                    if type(e) in [numpy.int32,numpy.int64]:
                        line.append(str(int(e)))
                    else:        
                        line.append(str(float(e)))
                elif at == 'string':
                    line.append(self.esc(e))
                elif at == 'date':
                    line.append(self.esc(e))
                elif at == 'nominal':
                    line.append(e)
                else:
                    raise Exception("Type " + at + " not supported for writing!")
            o.append(','.join(map(str, line)))
        return "\n".join(o) + "\n"

    def esc(self, s):
        "Escape a string if it contains spaces"
        try:
            if re.match(r'\s', s): return "\'" + str(s) + "\'"
        except TypeError:
            pass

        return str(s)

    def define_attribute(self, name, atype, data=None):
        """Define a new attribute. atype has to be one
        of 'numeric', 'string', 'date' or 'nominal'. For nominal
        or date attributes, pass the possible values as data."""
        n = self._rm_ticks(name)
        self.attributes.append(n)
        self.attribute_types[n] = atype
        self.attribute_data[n] = data

    def __parseline(self, l):
        if self.state == 'comment':
            if len(l) > 0 and l[0] == '%':
                self.comment.append(l[2:])
            else:
                self.comment = '\n'.join(self.comment)
                self.state = 'in_header'
                self.__parseline(l)
        elif self.state == 'in_header':
            ll = l.lower().strip()
            if ll.startswith('@relation '):
                self.__parse_relation(l)
            if ll.startswith('@attribute '):
                self.__parse_attribute(l)
            if ll.startswith('@data'):
                self.state = 'data'
        elif self.state == 'data':
            if len(l) > 0 and l[0] != '%':
                self.__parse_data(l)

    def __parse_relation(self, l):
        l = l.split()
        self.relation = l[1]

    def __parse_attribute(self, l):
        p = re.compile(r'[a-zA-Z_][a-zA-Z0-9_/+-.*\(\)]*|\{[^\}]+\}|\'[^\']+\'|\"[^\"]+\"')
        l = [s.strip() for s in p.findall(l)]
        name = l[1]
        atype = l[2]
        atypel = atype.lower()
        if (atypel == 'real' or
            atypel == 'numeric' or
            atypel == 'integer'):
            self.define_attribute(name, 'numeric')
        elif atypel == 'string':
            self.define_attribute(name, 'string')
        elif atypel == 'date':
            self.define_attribute(name, 'date', l[3])
        elif atype[0] == '{' and atype[-1] == '}':
            values = [s.strip () for s in atype[1:-1].split(',')]
            self.define_attribute(name, 'nominal', values)
        else:
            self.__print_warning("unsupported type " + atype + " for attribute " + name + ".")

    def __parse_data(self, line):
        l = [s.strip() for s in line.split(',')]
        if len(l) == 1:
            l = [s.strip() for s in line.split()]
        if not l[-1].strip(): # remove trailing empty item
            l.pop()
        if len(l) != len(self.attributes):
            self.__print_warning("contains wrong number of values")
            return

        datum = []
        for n, v in zip(self.attributes, l):
            at = self.attribute_types[n]
            if at == 'numeric':
                if v == '?' or v == '':
                    datum.append(numpy.nan)
                elif re.match(r'[+-]?[0-9]*(?:\.[0-9]*(?:[eE]-?[0-9]+)?)?', v):
                    try:    
                        datum.append(int(v))
                    except ValueError:
                        datum.append(float(v))
                else:
                    self.__print_warning('non-numeric value %s for numeric attribute %s' % (v, n))
                    return
            elif at == 'string':
                datum.append(self._rm_ticks(v))
            elif at == 'date':
                datum.append(self._rm_ticks(v))
            elif at == 'nominal':
                if v in self.attribute_data[n]:
                    datum.append(v)
                elif v == '?':
                    datum.append(numpy.nan)
                else:
                    self.__print_warning('incorrect value %s for nomial attribute %s' % (v, n))
                    return
        self.data.append(datum)

    def __print_warning(self, msg):
        #print ('Warning (line %d): ' % self.lineno) + msg
        pass

    def dump(self):
        """Print an overview of the ARFF file."""
        print("Relation " + self.relation)
        print("  With attributes")
        for n in self.attributes:
            if self.attribute_types[n] != 'nominal':
                print("    %s of type %s" % (n, self.attribute_types[n]))
            else:
                print(("    " + n + " of type nominal with values " +
                       ', '.join(self.attribute_data[n])))
        for d in self.data:
            print(d)
    


if __name__ == '__main__':
    if False:
        a = ArffFile.parse("""% yes
% this is great
@relation foobar
@attribute foo {a,b,c}
@attribute bar real
@data
a, 1
b, 2
c, d
d, 3
""")
        a.dump()

    a = ArffFile.load('../examples/diabetes.arff')

    print(a.write())
