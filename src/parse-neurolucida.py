"""
Parsing Neurolucida files.

Here we directly extract the tree and discard everything else.
It could easily be changed to extract labels and properties etc.

What could be done next is the creation of a Morphology object, possibly using labels.

Structure of Mustafa's files:
Axon ( Marker AIS (Marker Axon))
The marker is a short and thin piece of axon that has been manually added.
"""
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import (CharsNotIn, Optional, Suppress, Word, Regex, Combine, Group, delimitedList,
                       ParseException, alphas, nums, ZeroOrMore, Literal, Forward)
from os import listdir
from os.path import isfile, join


# Comments
COMMENT = Regex(r";.*").setName("Neurolucida comment")

# Terminal symbols
FLOAT = Combine(Word('+-'+nums,nums)+Literal('.')+Word(nums)).setParseAction(lambda t:float(t[0]))
INTEGER = Word('+-'+nums,nums).setParseAction(lambda t:int(t[0]))
NUMBER = FLOAT | INTEGER # this will always match floats
LABEL = Word(alphas, alphas+nums)
STRING = Suppress('"')+CharsNotIn('"')+Suppress('"')

# Basic elements
RGB = Suppress('RGB') + Suppress('(') + Group(INTEGER+Suppress(Optional(','))\
                + INTEGER+Suppress(Optional(','))\
                + INTEGER) + Suppress(')')

VALUE = NUMBER | STRING | RGB | LABEL

PROPERTY = Suppress('(') + LABEL + ZeroOrMore(VALUE) + Suppress(')')
POINT = Suppress("(") + Group(FLOAT + ZeroOrMore(Optional(Suppress(","))+FLOAT)) + Suppress(')')

# Tree
NODE = Forward()
END = Suppress(LABEL) | NODE.setResultsName('children')
BRANCH = Suppress(ZeroOrMore(STRING | PROPERTY)) + Group(ZeroOrMore(POINT)).setResultsName('branch') + END
NODE << Suppress('(') + Group(delimitedList(BRANCH,"|")) + Suppress(')')

# File
FILE = (Suppress(PROPERTY) + NODE).ignore(COMMENT)

def main():
    # Let's do it!
    ##path = "/Users/romain/Dropbox/Projects/Spike initiation/Collaborations/Maarten/Data/AIS reconstructions (.ASC)/"
    ##path1 = path+"Axo-dendritic/"
    ##path2 = path+"Axo-somatic/"
    ##filenames = [ (path1+f,f) for f in listdir(path1) if isfile(join(path1,f)) ] +\
    ##            [ (path2+f,f) for f in listdir(path2) if isfile(join(path2,f)) ]

    filenames = [
        ("../2013-07-29_#1-axodendritic.ASC", "2013-07-29_#1-axodendritic.ASC"),
        ("../2013-07-16_#1-axosomatic.ASC", "2013-07-16_#1-axosomatic.ASC")]


    #filename="2013-07-16_#1.asc"
    Ri = 150 / 100. # in Mohm.um

    for full_file_path,filename in filenames:
        handle_one_file(full_file_path, filename)
    plt.show()

def handle_one_file(file_path, filename):
    with open(file_path, "r") as f:
        text= f.read()
        parsed = FILE.parseString(text)

        axon_start = np.array(list(parsed[0]['branch'])).T
        AIS = np.array(list(parsed[0]['children'][1])).T
        AIS_start = axon_start.shape[1]
        axon = np.hstack((axon_start,AIS))
        cpt_length = sum(np.diff(axon[:3,:])**2,axis=0)**.5
        d = .5*(axon[3,:-1]+axon[3,1:])

        # Plotting
        plt.plot(np.cumsum(cpt_length[:AIS_start+1]),d[:AIS_start+1],'k')
        plt.plot(sum(cpt_length[:AIS_start])+np.cumsum(cpt_length[AIS_start:]),d[AIS_start:],'r')

        # Analysis
        AIS_onset = sum(sum(np.diff(np.array(list(parsed[0]['branch']))[:,:3].T)**2,axis=0)**.5)
        AIS_length = sum(sum(np.diff(np.array(list(parsed[0]['children'][1]))[:,:3].T)**2,axis=0)**.5) # should add length of first segment
        AIS_onset_Ra = 4/np.pi*np.Ri * sum(cpt_length[:AIS_start+1]/d[:AIS_start+1]**2)
        AIS_end_Ra = AIS_onset_Ra + 4/np.pi*np.Ri * sum((sum(np.diff(np.array(list(parsed[0]['children'][1]))[:,:3].T)**2,axis=0)**.5) /
                                    (np.array(list(parsed[0]['children'][1]))[:-1,3].T)**2)
        AIS_area = sum(cpt_length[AIS_start:]*np.pi*d[AIS_start:])

        # Ra calculated 5 um within the AIS
        n = AIS_start + np.where(np.cumsum(cpt_length[AIS_start:])>5.)[0][0]
        AIS_onset_Ra_5um = 4/np.pi*np.Ri * sum(cpt_length[:n]/d[:n]**2)

        print ( f"{filename}, {AIS_onset}, {AIS_length}, {AIS_onset_Ra}, {AIS_end_Ra}, {AIS_onset_Ra_5um}, {AIS_area}" )


main()

