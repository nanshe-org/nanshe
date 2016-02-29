"""
The package ``filter`` provides a variety of transformations for image data.

===============================================================================
Overview
===============================================================================
Provides a variety of different filters for data analysis. This includes
different types of convolutions or other related transformations. Operations on
masks are supported. Computations and removal of noise are also included.
Finally, support for performing wavelet transforms is provided.
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 31, 2015 22:29:31 EDT$"

__all__ = ["noise", "masks", "wavelet"]

from nanshe.imp.filters import masks
from nanshe.imp.filters import noise
from nanshe.imp.filters import wavelet
