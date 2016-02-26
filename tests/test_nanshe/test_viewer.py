__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 18, 2015 22:25:20 EDT$"


from sys import version_info

if version_info < (3,):
    import nanshe.viewer

del version_info
