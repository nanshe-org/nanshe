"""
The package ``hdf5`` provides support for IO operations on HDF5 files.

===============================================================================
Overview
===============================================================================
The package ``hdf5`` contains a variety helper functions and methods to address
IO operations on HDF5 files. Using this package requires a working copy of
|h5py|_ on your ``PYTHONPATH``.

.. |h5py| replace:: ``h5py``
.. _h5py: http://www.h5py.org/
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 27, 2015 09:32:02 EDT$"

__all__ = ["record", "search", "serializers"]

import record
import search
import serializers
