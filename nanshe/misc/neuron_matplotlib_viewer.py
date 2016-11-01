"""
The ``neuron_matplotlib_viewer`` module provides a ``matplotlib``-based viewer.

===============================================================================
Overview
===============================================================================
The module ``neuron_matplotlib_viewer`` provides a simple |matplotlib|_ viewer
for navigating through a 3D imagestack (TYX). However the first dimension could
be Z, as well. This has been deprecated in favor of the ``viewer``.

.. |matplotlib| replace:: ``matplotlib``
.. _matplotlib: http://matplotlib.org

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 03, 2014 20:20:39 EDT$"


import warnings

warnings.warn(
    "The module `neuron_matplotlib_viewer` is deprecated."
    "Please consider using `mplview` instead.",
    DeprecationWarning
)


from mplview.core import (
    MatplotlibViewer as NeuronMatplotlibViewer,
    SequenceNavigator as TimeNavigator,
)
