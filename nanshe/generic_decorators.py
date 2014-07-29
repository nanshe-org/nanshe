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


def static_variables(**kwargs):
    """
        Returns a decorator that decorates a callable such that it has the given static variables set.

        Args:
            *kwargs(tuple):     keyword args will be set to the value provided.

        Returns:
            (decorator):        a decorator for the callable.

    """

    def static_variables_tie(callable):
        """
            Decorates a function such that it has the given static variables set.

            Args:
                callable(callable):       the callable to decorate.

            Returns:
                (callable):               the callable returned.

        """

        for each_kwd, each_val in kwargs.items():
            setattr(callable, each_kwd, each_val)

        return(callable)

    return(static_variables_tie)