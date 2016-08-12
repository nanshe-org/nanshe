"""
``nanshe`` package, an image processing toolkit.

===============================================================================
Overview
===============================================================================

The ``nanshe`` package is an image processing package that contains a variety
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

===============================================================================
 Installation
===============================================================================

-------------------------------------------------------------------------------
 Dependencies
-------------------------------------------------------------------------------
Implementation of the algorithm has been done here in pure Python. However, a
few dependencies are required to get started. These include NumPy_, SciPy_,
h5py_, scikit-image_, SPAMS_, VIGRA_, and rank_filter_. The first 4 can be
found in standard distributions like Anaconda_. Installing VIGRA and
rank_filter can be done by using CMake_. SPAMS requires an existing BLAS/LAPACK
implementation. On Mac and Linux, this can be anything. Typically ATLAS_ is
used, but OpenBLAS_ or `Intel MKL`_ (if available) can be used, as well. This
will require modifying the setup.py script. On Windows, the setup.py links to
R_, which should be changed if another BLAS is available.

-------------------------------------------------------------------------------
 Building
-------------------------------------------------------------------------------
Python
===============================================================================
As this package is pure Python, building follows through the standard method.
Currently, we require setuptools_ for installation; so, make sure it is
installed. Then simply issue the following command to build and install.

.. code-block:: sh

    python setup.py install

Alternatively, one can build and then install in two steps if that is
preferable.

.. code-block:: sh

    python setup.py build
    python setup.py install

Conda
===============================================================================
Current packages can be found on our anaconda_ channel
( https://anaconda.org/nanshe/nanshe ). New ones are released every time a
passing tagged release is pushed to the ``master`` branch on GitHub. It is also
possible to build packages for conda_ for non-release commits as we do in our
continuous integration strategy.

To do this one requires the dependencies be installed or be available from a
anaconda channel. Additionally, one must be using the conda's ``root``
environment and have conda-build installed. Once this is done one need
only the run the following command with ``setup.py``.

.. code-block:: sh

    python setup.py bdist_conda

Assuming this completes successfully, conda will provide the path to the built
package.

-------------------------------------------------------------------------------
 Testing
-------------------------------------------------------------------------------
Running the test suite is fairly straightforward. Testing is done using nose_;
so, make sure you have a running copy if you wish to run the tests. Some of the
tests require drmaa_ installed and properly configured. If that is not the
case, those tests will be skipped automatically. To run the test suite, one
must be in the source directory. Then simply run the following command. This
will run all the tests and doctests. Depending on your machine, this will take
a few minutes to complete.

.. code-block:: sh

    nosetests

The full test suite includes 3D tests, which are very slow to run and so are
not run by default. As the code has been written to be dimensionally agnostic,
these tests don't cover anything that the 2D tests don't already cover. To run
the 3D tests, simply use ``setup.all.cfg``.

.. code-block:: sh

    nosetests -c setup.all.cfg

It is also possible to run this as part of the setup.py process. In which case,
this can be done as shown below. If 3D tests are required for this portion, one
need only replace ``setup.cfg`` with ``setup.all.cfg``.

.. code-block:: sh

    python setup.py nosetests

Also, the typical ``test`` subcommand can be used to run ``nosetests``, but no
other arguments are allowed.

.. code-block:: sh

    python setup.py test

-------------------------------------------------------------------------------
 Documentation
-------------------------------------------------------------------------------
Current documentation can be found on the GitHub page
( http://nanshe-org.github.io/nanshe/ ). A new copy is rebuilt any time there is
a passing commit is added to the ``master`` branch. Each documentation commit
is added to ``gh-pages`` branch with a reference to the commit in ``master``
that triggered the build as well as the tag (version) if provided.

It is also possible to build the documentation from source. This project uses
Sphinx_ for generating documentation. Please make sure you have it installed.
In particular, a version from 1.3 or later is required. Additionally, the
`Cloud Sphinx Theme`_ is required for generating the documentation and is used
in the HTML layout.

The ``rst`` files (outside of ``index.rst`` are not distributed with the source
code. This is because it is trivial to generate them and it is to easy for the
code to become out of sync with documentation if they are distributed. However,
building ``rst`` files has been made a dependency of all other documentation
build steps so one does not have to think about this. The preferred method for
building documentation is to use the ``setup.py`` hook as shown below. This
will build the RST files and place them in ``docs/``. It will also build the
HTML files by default and put them in the directory ``build/sphinx/html/``.
Simply open the ``index.html`` file to take a look.

.. code-block:: sh

    python setup.py build_sphinx

More build options can be determined by running the help command.

.. code-block:: sh

    python setup.py build_sphinx --help

-------------------------------------------------------------------------------
 Cleaning
-------------------------------------------------------------------------------
After any building operation a number of undesirable intermediate files are
created and left behind that one may wish to remove. To do this one merely
needs to run the clean command.

.. code-block:: sh

    python setup.py clean

This has been modified to also remove RST files generated when building
documentation. However, it will leave any final build products like HTML files.
If one wishes to remove everything built (including final build products), the
clean all command will do this.

.. code-block:: sh

    python setup.py clean --all

.. _`10.1109/ISBI.2013.6556660`: http://dx.doi.org/10.1109/ISBI.2013.6556660
.. _`10.1145/1553374.1553463`: http://dx.doi.org/10.1145/1553374.1553463
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _h5py: http://www.h5py.org/
.. _scikit-image: http://scikit-image.org/
.. _SPAMS: http://spams-devel.gforge.inria.fr/
.. _VIGRA: http://ukoethe.github.io/vigra/
.. _rank_filter: http://github.com/nanshe-org/rank_filter/
.. _Anaconda: http://store.continuum.io/cshop/anaconda/
.. _CMake: http://www.cmake.org/
.. _ATLAS: http://math-atlas.sourceforge.net/
.. _OpenBLAS: http://www.openblas.net/
.. _`Intel MKL`: http://software.intel.com/en-us/intel-mkl
.. _R: http://www.r-project.org/
.. _setuptools: http://pythonhosted.org/setuptools/
.. _anaconda: https://anaconda.org/
.. _conda: http://conda.pydata.org/
.. _nose: http://nose.readthedocs.org/en/latest/
.. _drmaa: http://github.com/pygridtools/drmaa-python
.. _Sphinx: http://sphinx-doc.org/
.. _`Cloud Sphinx Theme`: https://pythonhosted.org/cloud_sptheme/
"""
__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Dec 22, 2014 08:46:12 EST$"


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__all__ = [
    "box", "converter", "io", "imp", "learner", "registerer", "syn", "util"
]

from nanshe import box
from nanshe import converter
from nanshe import io
from nanshe import imp
from nanshe import learner
from nanshe import registerer
from nanshe import syn
from nanshe import util
