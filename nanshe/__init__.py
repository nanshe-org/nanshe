"""
The ```nanshe``` package is an image processing package that contains a variety
of different techniques, which are used primarily to assemble the ADINA
algorithm proposed by Diego, et al.
( http://dx.doi.org/10.1109/ISBI.2013.6556660 ) to extract active neurons from
an image sequence. This algorithm uses online dictionary learning (a form of
matrix factorization) at its heart as implemented by Marial, et al.
( http://dx.doi.org/10.1145/1553374.1553463 ) to find a set of atoms (or basis
images) that are representative of an image sequence and can be used to
approximately reconstruct the sequence. However, it is designed in a modular
way so that a different matrix factorization could be swapped in and
appropriately parameterized. Other portions of the algorithm include a
preprocessing phase that has a variety of different techniques that can be
applied optionally. For example, removing registration artifacts from
a line-by-line registration algorithm, background subtraction, and a wavelet
transform to filter objects in a particular size.
"""
__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Dec 22, 2014 08:46:12 EST$"


__all__ = [
    "converter", "io", "imp", "learner" "registerer", "spams_sandbox",
    "synthetic_data", "util",  # "viewer"
]

import converter
import io
import imp
import learner
import registerer
import spams_sandbox
import synthetic_data
import util
# import viewer           # Must be commented as there is some segfault coming from Volumina.
