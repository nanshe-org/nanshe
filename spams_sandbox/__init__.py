"""
    SPAMS seems to step on the interpreter. Despite our best efforts to sandbox it, in a separate thread, it manages
    to still get messed up when neuron_volumina_viewer is on the path. Therefore, we will place all access to it in a
    separate module that cannot see the contents of nanshe. Hopefully, this will make it less likely to create a
    segmentation fault.
"""

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 20, 2014 12:01:08 EDT$"


__all__ = ["..", "spams_sandbox"]


import spams_sandbox