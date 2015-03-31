"""
The ```nanshe``` package is an image processing package that contains a variety
of different techniques, which are used primarily to assemble the ADINA
algorithm proposed by Diego, et al.
( doi:`10.1109/ISBI.2013.6556660`_ ) to extract active neurons from
an image sequence. This algorithm uses online dictionary learning (a form of
matrix factorization) at its heart as implemented by Marial, et al.
( doi:`10.1145/1553374.1553463`_ ) to find a set of atoms (or basis
images) that are representative of an image sequence and can be used to
approximately reconstruct the sequence. However, it is designed in a modular
way so that a different matrix factorization could be swapped in and
appropriately parameterized. Other portions of the algorithm include a
preprocessing phase that has a variety of different techniques that can be
applied optionally. For example, removing registration artifacts from
a line-by-line registration algorithm, background subtraction, and a wavelet
transform to filter objects in a particular size.

Implementation of the algorithm has been done here in pure Python. However, a
few dependencies are required to get started. These include NumPy_, SciPy_,
h5py_, scikit-image_, SPAMS_, VIGRA_, and rank_filter_. The first 4 can be
found in standard distributions like Anaconda_. Installing VIGRA and
rank_filter can be done by using CMake_. SPAMS requires an existing BLAS/LAPACK
implementation. On Mac and Linux, this can be anything. Typically ATLAS_ is
used, but OpenBLAS_ or `Intel MKL`_ (if available) can be used, as well. This
will require modifying the setup.py script. On Windows, the setup.py links to
R_, which should be changed if another BLAS is available.


.. _`10.1109/ISBI.2013.6556660`: http://dx.doi.org/10.1109/ISBI.2013.6556660
.. _`10.1145/1553374.1553463`: http://dx.doi.org/10.1145/1553374.1553463
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _h5py: http://www.h5py.org/
.. _scikit-image: http://scikit-image.org/
.. _SPAMS: http://spams-devel.gforge.inria.fr/
.. _VIGRA: http://ukoethe.github.io/vigra/
.. _rank_filter: https://github.com/jakirkham/rank_filter/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _CMake: http://www.cmake.org/
.. _ATLAS: http://math-atlas.sourceforge.net/
.. _OpenBLAS: http://www.openblas.net/
.. _`Intel MKL`: https://software.intel.com/en-us/intel-mkl
.. _R: http://www.r-project.org/
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
