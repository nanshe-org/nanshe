|Travis Build Status| |Wercker Build Status| |Coverage Status| |Code Health| |License| |Documentation| |Anaconda Release| |Gitter|

--------------

Nanshe
======

Motivation
----------

This package is designed to perform segmentation on datasets containing
spatially-fixed components with time varying intensity. As the
techniques used are a bit slow, the segmentation algorithm has been
designed to run in batch. The segmentation algorithm has been further
tailored to enable multiprocessing as well as support for clusters that
support DRMAA ( http://www.drmaa.org ). In addition to the segmentation
algorithm, it provides a converter for TIFF to HDF5 and a rough viewer
for inspecting the results obtained.

Converter
---------

Intro
~~~~~

The first item contained inside the package is ``nanshe_converter.py``.
This is designed to convert external formats to HDF5 (the native format
used by Nanshe). Currently, only TIFF stacks are supported.

Sample Usage
~~~~~~~~~~~~

To run the converter, simply run the command below. The first argument
specifies the input format. Currently, this is just tiff. The
configuration parameters are contained in the ``config.json`` file. An
example configuration file can be found in the ``examples`` folder. The
input images can be specified using a regex as can be seen below. The
output file is an HDF5 with a dataset containing the result. Here,
``output.h5`` is the output file. The data is stored in the specified
dataset, which is ``/images`` within root.

::

    ./nanshe_converter.py tiff config.json input_dir/*.tiff output.h5/images

Learner
-------

Intro
~~~~~

The next item contained inside the package is ``nanshe_learner.py``.
This runs the segmentation algorithm using a specified set of parameters
on a provided dataset and stores the result.

Sample Usage
~~~~~~~~~~~~

To run the learner, simply run the command below. The configuration
parameters are contained in the ``config.json`` file. Sample
configuration files are included in the ``examples`` folder. One example
is for a single process and is called ``nanshe_learner.json``. Another
example uses multiprocessing and is called
``nanshe_learner_multiprocessing.json``. The input data is stored within
the HDF5 ``input.h5`` in the top level group in the dataset ``images``.
The output file can be the same file or a different file. Here,
``output.h5`` is the output file. The data is stored in the specified
group, which is root as can be seen by the following ``/``. The
extracted neurons will be put in a compound data type (a.k.a. a
structured array) called ``neurons``. Various results including the
masks (under the field ``mask``) will be stored here. The first index on
neurons will indicate which neuron is selected. All remaining indices
for different types follow C-order convention (excepting
``gaussian_cov``, which is the covariance matrix and slightly differs
for obvious reasons).

::

    ./nanshe_learner.py config.json input.h5/images output.h5/

Viewer
------

Intro
~~~~~

The first item contained inside the package is ``nanshe_viewer.py``.
This runs the segmentation algorithm using a specified set of parameters
on a provided dataset and stores the result.

Sample Usage
~~~~~~~~~~~~

To run the viewer, simply run the command below. The configuration
parameters are contained in the ``config.json`` file. These specify how
the layers are ordered, grouped, and named. Also, they specify where to
fetch the data for the layers. Lastly, they specify where this data is
located in an HDF5 file. The input data is stored within the HDF5
``input.h5``.

::

    ./nanshe_viewer.py config.json input.h5

Prebuilt Binaries
-----------------

There are some prebuilt binaries that are available. These are built
using BuildEM ( https://github.com/janelia-flyem/buildem ). The binaries
are designed to be largely independent of your own system. These contain
all the dependencies needed to run everything used by Nanshe except for
the ones explicitly required and excluded from BuildEM.

To run any of the above utilities, there is a ``run_*.sh`` command that
will do this. For example, here is how you would call the
nanshe\_learner.

::

    ./nanshe-package/run_nanshe_learner.sh "config.json" "input.h5/images" "output.h5/"

It is also possible to submit jobs to cluster using the Open Grid
Engine, which have DRMAA installed. This is done using ``qrun_*.sh``
commands as seen below. In the case of nanshe\_learner, a special
configure file is required, which can be found in the example directory.
If DRMAA is not available, it is possible to run multiprocessing as
mentioned above.

::

    ./nanshe-package/qrun_nanshe_learner.sh "config.json" "input.h5/images" "output.h5/"

Conventions
-----------

Throughout nanshe, there are certain conventions that are followed.
Often it is not necessary to know them. However, they do occasionally
come up. For instances, Nanshe uses the C-order indexing convention and
does not use multiple channels. This means that the spatial ordering
will be transposed of what is usually expected and time will come first.
For a given 2D array, ``a_2D``, indexing would be ``a_2D[t,y,x]`` where
``t`` is time and ``x``, ``y`` are the usual spatial coordinates.
Similarly, for a 3D array, ``a_3D``, the indexing would be
``a_3D[t,z,y,x]``. Also, the JSON configuration files permit commenting.
This is done by using ``__comment__`` as a prefix within a string. For
objects (a.k.a. dictionaries), if the key or value has a comment both
will be removed on parsing. For arrays (a.k.a. lists), if a value is a
comment, it will be dropped from the list. Additionally, some steps are
optional, in which case they can be removed or commented from the
configuration file. For certain fields, like those in
"accepted\_region\_shape\_constraints" and
"accepted\_neuron\_shape\_constraints", it is permissible to have a
"min" and/or "max". If both are specified, then the range must be met.
If only one is specified, then only an upper or lower bound exists.

.. |Travis Build Status| image:: https://travis-ci.org/nanshe-org/nanshe.svg?branch=master
   :target: https://travis-ci.org/nanshe-org/nanshe
.. |Wercker Build Status| image:: https://app.wercker.com/status/0a126e02ea98f90a54a43fa12d20570a/s/master
   :target: https://app.wercker.com/project/bykey/0a126e02ea98f90a54a43fa12d20570a
.. |Coverage Status| image:: https://coveralls.io/repos/nanshe-org/nanshe/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/nanshe-org/nanshe?branch=master
.. |Code Health| image:: https://landscape.io/github/nanshe-org/nanshe/master/landscape.svg?style=flat
   :target: https://landscape.io/github/nanshe-org/nanshe/master
.. |License| image:: https://img.shields.io/github/license/nanshe-org/nanshe.svg
   :target: ./LICENSE.txt
.. |Documentation| image:: https://img.shields.io/badge/docs-current-9F21E9.svg
   :target: http://nanshe-org.github.io/nanshe/
.. |Anaconda Release| image:: https://anaconda.org/nanshe/nanshe/badges/version.svg
   :target: https://anaconda.org/nanshe/nanshe
.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/nanshe-org/nanshe
   :target: https://gitter.im/nanshe-org/nanshe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
