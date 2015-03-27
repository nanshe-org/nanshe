__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 26, 2014 18:24:25 EDT$"


import glob

import prof



# Get the logger
logger = prof.logging.getLogger(__name__)



@prof.log_call(logger)
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
