"""
The module ``wrappers`` provides support decorating functions and classes.

===============================================================================
Overview
===============================================================================
The module ``wrappers`` extends wrapping abilities found in |functools|_. In
particular, it is ensured all wrapped functions contain an attribute
``__wrapped__``, which points back to the original function before the wrapper
was applied. Also, the ability to wrap classes with a decorator to apply a
``metaclass`` or series of ``metaclass``es is provided. Making it much easier
to transform classes without mucking in their internals. For classes inheriting
from ``Qt`` objects a special decorator is provided to make sure they
participate in the correct inheritance scheme.

.. |functools| replace:: ``functools``
.. _functools: http://docs.python.org/2/library/functools.html

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 23, 2014 16:24:36 EDT$"


import collections
import inspect
import itertools
import functools
import types

import PyQt4.QtCore


def update_wrapper(wrapper,
                   wrapped,
                   assigned=functools.WRAPPER_ASSIGNMENTS,
                   updated=functools.WRAPPER_UPDATES):
    """
        Extends functools.update_wrapper to ensure that it stores the wrapped
        function in the attribute __wrapped__.

        Args:
            wrapper(callable):      the replacement callable.

            wrapped(callable):      the callable that is being wrapped.

            assigned(tuple):        is a tuple naming the attributes assigned
                                    directly from the wrapped function to the
                                    wrapper function (defaults to
                                    functools.WRAPPER_ASSIGNMENTS)

            updated(tuple):         is a tuple naming the attributes of the
                                    wrapper that are updated with the
                                    corresponding attribute from the wrapped
                                    function (defaults to
                                    functools.WRAPPER_UPDATES)

        Returns:
            (callable):             the wrapped callable.
    """

    wrapper = functools.update_wrapper(
        wrapper, wrapped, assigned=assigned, updated=updated
    )

    # Store the underlying callable. Automatic in Python 3.
    setattr(wrapper, "__wrapped__", getattr(wrapper, "__wrapped__", wrapped))

    return(wrapper)


def wraps(wrapped,
          assigned=functools.WRAPPER_ASSIGNMENTS,
          updated=functools.WRAPPER_UPDATES):
    """
        Builds on functools.wraps to ensure that it stores the wrapped function
        in the attribute __wrapped__.

        Args:
            wrapped(callable):      the callable that is being wrapped.

            assigned(tuple):        is a tuple naming the attributes assigned
                                    directly from the wrapped function to the
                                    wrapper function (defaults to
                                    functools.WRAPPER_ASSIGNMENTS)

            updated(tuple):         is a tuple naming the attributes of the
                                    wrapper that are updated with the
                                    corresponding attribute from the wrapped
                                    function (defaults to
                                    functools.WRAPPER_UPDATES)

        Returns:
            (callable):             a decorator for callable, which will
                                    contain wrapped.
    """

    return(functools.partial(
        update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
    ))


def identity_wrapper(a_callable):
    """
        Trivially wraps a given callable without doing anything else to it.

        Args:
            a_callable(callable):   the callable that is being wrapped.

        Returns:
            (callable):             a wrapped callable.
    """

    @wraps(a_callable)
    def wrapped_callable(*args, **kwargs):
        """
            Trivially wraps a given callable without doing anything else to it.

            Args:
                *args:      Variable length argument list.
                **kwargs:   Arbitrary keyword arguments.

            Returns:
                Same as what `a_callable` returns.
        """

        return(a_callable(*args, **kwargs))

    return(wrapped_callable)


def static_variables(**kwargs):
    """
        Returns a decorator that decorates a callable such that it has the
        given static variables set.

        Args:
            *kwargs(tuple):     keyword args will be set to the value provided.

        Returns:
            (decorator):        a decorator for the callable.
    """

    def static_variables_tie(a_callable):
        """
            Decorates a function such that it has the given static variables
            set.

            Args:
                a_callable(callable):     the callable to decorate.

            Returns:
                (callable):               the callable returned.

        """

        callable_wrapped = identity_wrapper(a_callable)

        for each_kwd, each_val in kwargs.items():
            setattr(callable_wrapped, each_kwd, each_val)

        return(callable_wrapped)

    return(static_variables_tie)


def metaclass(meta):
    """
        Returns a decorator that decorates a class such that the given
        metaclass is applied.

        Note:
            Decorator will add the __metaclass__ attribute so the last
            metaclass applied is known. Also, decorator will add the
            __wrapped__ attribute so that the unwrapped class can be retrieved.

        Args:
            meta(metaclass):     metaclass to apply to a given class.

        Returns:
            (decorator):         a decorator for the class.
    """

    def metaclass_wrapper(cls):
        """
            Returns a decorated class such that the given metaclass is applied.

            Note:
                Adds the __metaclass__ attribute so the last metaclass used is
                known. Also, adds the __wrapped__ attribute so that the
                unwrapped class can be retrieved.

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
        Returns a decorator that decorates a class such that the given
        metaclasses are applied.

        Note:
            Shorthand for repeated application of metaclass.

        Args:
            *metas(metaclasses):     metaclasses to apply to a given class.

        Returns:
            (decorator):             a decorator for the class.
    """

    def metaclasses_wrapper(cls):
        """
            Returns a decorated class such that the given metaclasses are
            applied.

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
        Returns a decorator that decorates a class such that it has the given
        static variables set.

        Args:
            **kwargs(tuple):     keyword args will be set to the value
                                 provided.

        Returns:
            (decorator):         a decorator for the class.
    """

    class MetaStaticVariables(type):
        """
            Metaclass, which adds static variable with the given value to a
            class.
        """

        def __new__(meta, name, bases, dct):
            dct.update(kwargs)

            return(super(MetaStaticVariables, meta).__new__(
                meta, name, bases, dct
            ))

    return(metaclass(MetaStaticVariables))


def class_decorate_all_methods(*decorators):
    """
        Returns a decorator that decorates a class such that all its methods
        are decorated by the decorators provided.

        Args:
            *decorators(tuple):     decorators to decorate all methods with.

        Returns:
            (decorator):            a decorator for the class.
    """

    class MetaAllMethodsDecorator(type):
        """
            Metaclass, which decorates all methods with the list of decorators
            in order.
        """

        def __new__(meta, name, bases, dct):
            for _k, _v in dct.items():
                # Are all of FunctionType at this point.
                # Will be of MethodType at a later step.
                if isinstance(_v, types.FunctionType):
                    for each_decorator in decorators:
                        _v = each_decorator(_v)

                dct[_k] = _v

            return(super(MetaAllMethodsDecorator, meta).__new__(
                meta, name, bases, dct
            ))

    return(metaclass(MetaAllMethodsDecorator))


def qt_class_decorate_all_methods(*decorators):
    """
        Returns a decorator that decorates a class such that all its methods
        are decorated by the decorators provided.

        Args:
            *decorators(tuple):     decorators to decorate all methods with.

        Returns:
            (decorator):            a decorator for the class.
    """

    class MetaAllMethodsDecorator(PyQt4.QtCore.pyqtWrapperType):
        """
            Metaclass, which decorates all methods with the list of decorators
            in order.

            Inherits from PyQt4.QtCore.pyqtWrapperType based on this
            ( http://www.gulon.co.uk/2012/12/28/pyqt4-qobjects-and-metaclasses/ ).
        """

        def __new__(meta, name, bases, dct):
            for _k, _v in dct.items():
                # Are all of FunctionType at this point.
                # Will be of MethodType at a later step.
                if isinstance(_v, types.FunctionType):
                    for each_decorator in decorators:
                        _v = each_decorator(_v)

                dct[_k] = _v

            return(super(MetaAllMethodsDecorator, meta).__new__(
                meta, name, bases, dct
            ))

    return(metaclass(MetaAllMethodsDecorator))


def class_decorate_methods(**method_decorators):
    """
        Returns a decorator that decorates a class such that specified methods
        are decorated by the decorators provided.

        Args:
            **method_decorators(tuple):     method names with a single
                                            decorator or a list of decorators.

        Returns:
            (decorator):                    a decorator for the class.
    """

    class MetaMethodsDecorator(type):
        """
            Metaclass, which decorates some methods based on the keys given.
            Uses the decorator(s) provided for each method to decorator in
            order.
        """

        def __new__(meta, name, bases, dct):
            for _k, _v in dct.items():
                if isinstance(_v, types.FunctionType):
                    _dl = method_decorators.get(_k)
                    if (_dl is not None):
                        try:
                            iter(_dl)
                        except TypeError:
                            _dl = [_dl]

                        for _d in _dl:
                            _v = _d(_v)

                dct[_k] = _v

            return(super(MetaMethodsDecorator, meta).__new__(
                meta, name, bases, dct
            ))

    return(metaclass(MetaMethodsDecorator))


def unwrap(a_callable):
    """
        Returns the underlying function that was wrapped.

        Args:
            a_callable(callable):     some wrapped (or not) callable.

        Returns:
            (callable):               the callable that is no longer wrapped.
    """

    unwrapped_callable = a_callable

    while hasattr(unwrapped_callable, "__wrapped__"):
        unwrapped_callable = unwrapped_callable.__wrapped__

    return(unwrapped_callable)


def tied_call_args(a_callable, *args, **kwargs):
    """
        Ties all the args to their respective variable names.

        Args:
            a_callable(callable):     some callable.
            *args(callable):          positional arguments for the callable.
            **kwargs(callable):       keyword arguments for the callable.

        Returns:
            args (tuple):             ordered dictionary of arguments name and
                                      their values, all variadic position
                                      arguments, all variadic keyword
                                      arguments.
    """

    sig = inspect.getargspec(a_callable)

    unsorted_callargs = inspect.getcallargs(a_callable, *args, **kwargs)

    new_args = tuple()
    if (sig.varargs is not None):
        new_args = unsorted_callargs[sig.varargs]

    new_kwargs = dict()
    if (sig.keywords is not None):
        new_kwargs = unsorted_callargs[sig.keywords]

    callargs = collections.OrderedDict()
    for each_arg in sig.args:
        callargs[each_arg] = unsorted_callargs[each_arg]

    return(callargs, new_args, new_kwargs)


def repack_call_args(a_callable, *args, **kwargs):
    """
        Reorganizes args and kwargs to match the given callables signature.

        Args:
            a_callable(callable):     some callable.
            *args(callable):          positional arguments for the callable.
            **kwargs(callable):       keyword arguments for the callable.

        Returns:
            args (tuple):             all arguments as passed as position
                                      arguments, all default arguments and
                                      all arguments passed as keyword
                                      arguments.
    """

    callargs, new_args, new_kwargs = tied_call_args(
        a_callable, *args, **kwargs
    )

    new_args = tuple(callargs.values()[:len(args)]) + new_args
    new_kwargs.update(dict(callargs.items()[len(args):]))

    return(new_args, new_kwargs)


def with_setup_state(setup=None, teardown=None):
    """
        Adds setup and teardown callable to a function s.t. they can mutate it.

        Based on ``with_setup`` from ``nose``. This goes a bit further than
        ``nose`` does and provides a mechanism for the setup and teardown
        functions to change the callable in question. In other words, variables
        generated in setup can be stored in the functions globals and then
        cleaned up and removed in teardown. The final result of using this
        function should be a function equivalent to one generated by
        ``with_setup``.

        Args:
            setup(callable):        A callable that takes the decorated
                                    function as an argument. This sets up the
                                    function before execution.

            teardown(callable):     A callable that takes the decorated
                                    function as an argument. This cleans up the
                                    function after execution.

        Returns:
            callable:               Does the actual decoration.
    """

    def with_setup_state_wrapper(a_callable, setup=setup, teardown=teardown):
        """
            Mutates the callable s.t. it has globals for setup and teardown.

            Args:
                setup(callable):        A callable that takes the decorated
                                        function as an argument. This sets up
                                        the function before execution. Simply
                                        forwarded from before.

                teardown(callable):     A callable that takes the decorated
                                        function as an argument. This cleans up
                                        the function after execution. Simply
                                        forwarded from before.

            Returns:
                callable:               The original callable with setup and
                                        teardown globals.
        """

        stage_dict = {"setup": setup, "teardown": teardown}
        stage_orderer = [(lambda a, b: (a, b)), (lambda a, b: (b, a))]
        stage_itr = itertools.izip(stage_dict.items(), stage_orderer)

        for (each_stage_name, each_new_stage), each_stage_orderer in stage_itr:
            each_old_stage = getattr(a_callable, each_stage_name, None)
            if each_new_stage:
                each_new_stage = functools.partial(each_new_stage, a_callable)

            chained_stages = None
            if each_old_stage and each_new_stage:
                first_stage, second_stage = each_stage_orderer(
                    each_old_stage, each_new_stage
                )

                def stages(first_stage=first_stage,
                           second_stage=second_stage):
                    first_stage()
                    second_stage()

                chained_stages = stages
            elif each_old_stage:
                chained_stages = each_old_stage
            elif each_new_stage:
                def a_stage(each_new_stage=each_new_stage):
                    each_new_stage()

                chained_stages = a_stage
            else:
                continue

            setattr(a_callable, each_stage_name, chained_stages)

        return(a_callable)

    return(with_setup_state_wrapper)
