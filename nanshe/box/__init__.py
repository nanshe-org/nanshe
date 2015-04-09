"""
The ``box`` package exists to hold sandboxed versions of algorithms.

===============================================================================
Overview
===============================================================================
In particular, SPAMS sometimes seems to step on the interpreter. In general,
the strategy is to launch it in a separate process so that it hopefully does
not mess up the main interpreter. More details of the strategies used for SPAMS
can be found in :py:mod:`~nanshe.box.spams_sandbox`.
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 20, 2014 12:01:08 EDT$"


__all__ = ["spams_sandbox"]


import spams_sandbox
