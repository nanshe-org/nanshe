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


def metaclass(meta):
    """
        Returns a decorator that decorates a class such that the given metaclass is applied.

        Note:
            Decorator will add the __metaclass__ attribute so the last metaclass applied is known.
            Also, decorator will add the __wrapped__ attribute so that the unwrapped class can be retrieved.

        Args:
            meta(metaclass):     metaclass to apply to a given class.

        Returns:
            (decorator):         a decorator for the class.

    """

    def metaclass_wrapper(cls):
        """
            Returns a decorated class such that the given metaclass is applied.

            Note:
                Adds the __metaclass__ attribute so the last metaclass used is known.
                Also, adds the __wrapped__ attribute so that the unwrapped class can be retrieved.

            Args:
                cls(class):          class to decorate.

            Returns:
                (class):             the decorated class.

        """

        __name = str(cls.__name__)
        __bases = tuple(cls.__bases__)
        __dict = dict(cls.__dict__)

        __dict.pop("__dict__", None)
        __dict.pop("__weakref__", None)

        for each_slot in __dict.get("__slots__", tuple()):
            __dict.pop(each_slot, None)

        __dict["__metaclass__"] = meta
        __dict["__wrapped__"] = cls

        return(meta(__name, __bases, __dict))

    return(metaclass_wrapper)


def metaclasses(*metas):
    """
        Returns a decorator that decorates a class such that the given metaclasses are applied.

        Note:
            Shorthand for repeated application of metaclass.

        Args:
            *metas(metaclasses):     metaclasses to apply to a given class.

        Returns:
            (decorator):             a decorator for the class.

    """

    def metaclasses_wrapper(cls):
        """
            Returns a decorated class such that the given metaclasses are applied.

            Args:
                cls(class):          class to decorate.

            Returns:
                (class):             the decorated class.

        """

        new_cls = cls

        for each_meta in metas:
            new_cls = metaclass(each_meta)(new_cls)

        return(new_cls)

    return(metaclasses_wrapper)


def class_static_variables(**kwargs):
    """
        Returns a decorator that decorates a class such that it has the given static variables set.

        Args:
            **kwargs(tuple):     keyword args will be set to the value provided.

        Returns:
            (decorator):         a decorator for the class.

    """

    class MetaStaticVariables(type):
        """
            Metaclass, which adds static variable with the given value to a class.
        """

        def __new__(meta, name, bases, dct):
            dct.update(kwargs)

            return(super(MetaStaticVariables, meta).__new__(meta, name, bases, dct))

    return(metaclass(MetaStaticVariables))
