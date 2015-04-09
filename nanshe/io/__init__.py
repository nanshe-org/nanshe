"""
The package ``io`` provides support for various IO operations.

===============================================================================
Overview
===============================================================================
The package ``io`` contains a variety helper functions and methods to address
IO operations. Particular focus is given to HDF5. Support is also provided in
using JSON as a configuration format and conversion of TIFF to HDF5.
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 29, 2015 23:03:14 EDT$"

__all__ = ["hdf5", "xjson", "xtiff"]

import hdf5
import xjson
import xtiff
