"""
The package ``misc`` contains everything that does not really belong elsewhere.

===============================================================================
Overview
===============================================================================
The package ``misc`` (short for **misc**\ ellaneous) contains random components
that did not really belong anywhere else. This stuff is likely not well
documented or tested. As a consequence, it is safe to assume that this stuff
is not stable and may be deprecated. If does ever leave this package (and is
not deleted), its API may be broken from what it was here.
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 01, 2015 00:13:16 EDT$"

__all__ = [  # "neuron_matplotlib_viewer",
    "random_dictionary_learning_data"
]

# from nanshe.misc import neuron_matplotlib_viewer
from nanshe.misc import random_dictionary_learning_data
