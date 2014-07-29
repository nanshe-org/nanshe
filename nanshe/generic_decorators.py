__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 23, 2014 16:24:36 EDT$"


import functools


def update_wrapper(wrapper,
                   wrapped,
                   assigned = functools.WRAPPER_ASSIGNMENTS,
                   updated = functools.WRAPPER_UPDATES):
    """
        Extends functools.update_wrapper to ensure that it stores the wrapped function in the attribute __wrapped__.

        Args:
            wrapper(callable):      the replacement callable.

            wrapped(callable):      the callable that is being wrapped.

            assigned(tuple):        is a tuple naming the attributes assigned directly
                                    from the wrapped function to the wrapper function (defaults to
                                    functools.WRAPPER_ASSIGNMENTS)

            updated(tuple):         is a tuple naming the attributes of the wrapper that
                                    are updated with the corresponding attribute from the wrapped
                                    function (defaults to functools.WRAPPER_UPDATES)

        Returns:
            (callable):             the wrapped callable.
    """

    wrapper = functools.update_wrapper(wrapper, wrapped, assigned = assigned, updated = updated)

    # Store the underlying callable. Automatic in Python 3.
    setattr(wrapper, "__wrapped__", wrapped)

    return(wrapper)


def wraps(wrapped,
          assigned = functools.WRAPPER_ASSIGNMENTS,
          updated = functools.WRAPPER_UPDATES):
    """
        Builds on functools.wraps to ensure that it stores the wrapped function in the attribute __wrapped__.

        Args:
            wrapped(callable):      the callable that is being wrapped.

            assigned(tuple):        is a tuple naming the attributes assigned directly
                                    from the wrapped function to the wrapper function (defaults to
                                    functools.WRAPPER_ASSIGNMENTS)

            updated(tuple):         is a tuple naming the attributes of the wrapper that
                                    are updated with the corresponding attribute from the wrapped
                                    function (defaults to functools.WRAPPER_UPDATES)

        Returns:
            (callable):             a decorator for callable, which will contain wrapped.
    """

    return(functools.partial(update_wrapper, wrapped = wrapped, assigned = assigned, updated = updated))