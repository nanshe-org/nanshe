"""
The package ``imp`` contains algorithms useful for image processing.

===============================================================================
Overview
===============================================================================
Algorithms in the ``imp`` package include filters/convolutions, registration,
and segmentation algorithms. Together these algorithms are composed to provide
workflows for analyzing image data start to finish.
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 14, 2014 20:37:08 EDT$"

__all__ = ["filters", "registration", "renorm", "segment"]

from nanshe.imp import filters
from nanshe.imp import registration
from nanshe.imp import renorm
from nanshe.imp import segment
