"""
The module ``xglob`` extends the abilities found in Python's ``glob``.

===============================================================================
Overview
===============================================================================
The module ``xglob`` extends the abilities found in |glob|_. In particular, it
provides a function for iterating through several glob expressions and joining
them together.

.. |glob| replace:: ``glob``
.. _glob: http://docs.python.org/2/library/glob.html

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 26, 2014 18:24:25 EDT$"


import glob

import prof



# Get the logger
trace_logger = prof.getTraceLogger(__name__)



@prof.log_call(trace_logger)
def expand_pathname_list(*pathnames):
    """
        Takes each pathname in those given and expands them using regex.

        Args:
            *pathnames(str):     pathnames to use regex on to expand.

        Returns:
            list:                a list of path names (without regex)
    """

    expanded_pathnames = []

    # Completes any regex
    expanded_pathnames = []
    for each_pathname in pathnames:
        expanded_pathnames.extend(glob.glob(each_pathname))

    return(expanded_pathnames)
