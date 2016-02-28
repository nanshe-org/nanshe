from __future__ import print_function


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 25, 2015 13:30:52 EDT$"


import functools
import sys

import PyQt4
import PyQt4.QtCore

import nanshe.util.wrappers


class TestWrappers(object):
    def test_update_wrapper(self):
        def wrapper(a_callable):
            def wrapped(*args, **kwargs):
                return(a_callable(*args, **kwargs))

            return(wrapped)

        def func(a, b=2):
            return(a + b)

        func_wrapped_1 = functools.update_wrapper(wrapper, func)
        if not hasattr(func_wrapped_1, "__wrapped__"):
            setattr(func_wrapped_1, "__wrapped__", func)

        func_wrapped_2 = nanshe.util.wrappers.update_wrapper(
            wrapper, func
        )

        assert func_wrapped_1 == func_wrapped_2


    def test_wraps(self):
        def wrapper(a_callable):
            def wrapped(*args, **kwargs):
                return(a_callable(*args, **kwargs))

            return(wrapped)

        def func(a, b=2):
            return(a + b)

        func_wrapped_1 = functools.wraps(wrapper)(func)
        if not hasattr(func_wrapped_1, "__wrapped__"):
            setattr(func_wrapped_1, "__wrapped__", func)

        func_wrapped_2 = nanshe.util.wrappers.wraps(wrapper)(
            func
        )

        assert func_wrapped_1 == func_wrapped_2


    def test_identity_wrapper(self):
        def func(a, b=2):
            return(a + b)

        func_wrapped = nanshe.util.wrappers.identity_wrapper(
            func
        )

        assert func_wrapped != func
        assert not hasattr(func, "__wrapped__")
        assert hasattr(func_wrapped, "__wrapped__")
        assert func_wrapped.__wrapped__ == func


    def test_static_variables(self):
        def func(a, b=2):
            return(a + b)

        func_wrapped = nanshe.util.wrappers.static_variables(
            c=7
        )(
            func
        )

        assert func_wrapped.__wrapped__ == func
        assert not hasattr(func, "c")
        assert hasattr(func_wrapped, "c")
        assert func_wrapped.c == 7


    def test_metaclass_0(self):
        class Meta(type):
            pass

        class Class(object):
            pass

        ClassWrapped = nanshe.util.wrappers.metaclass(Meta)(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class


    def test_metaclass_1(self):
        class Meta(type):
            pass

        class Class(object):
            __slots__ = ("__special_object__",)

            def __init__(self):
                self.__special_object__ = object

        ClassWrapped = nanshe.util.wrappers.metaclass(Meta)(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class

        a = Class()
        b = ClassWrapped()

        assert hasattr(a, "__special_object__")
        assert hasattr(b, "__special_object__")
        assert b.__special_object__ == a.__special_object__


    def test_metaclasses_0(self):
        class Meta1(type):
            pass

        class Meta2(type):
            pass

        class Class(object):
            pass

        ClassWrapped = nanshe.util.wrappers.metaclasses(
            Meta1, Meta2
        )(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert hasattr(ClassWrapped.__wrapped__, "__wrapped__")
        assert ClassWrapped.__wrapped__.__wrapped__ == Class


    def test_metaclasses_1(self):
        class Meta1(type):
            pass

        class Meta2(type):
            pass

        class Class(object):
            __slots__ = ("__special_object__",)

            def __init__(self):
                self.__special_object__ = object

        ClassWrapped = nanshe.util.wrappers.metaclasses(
            Meta1, Meta2
        )(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert hasattr(ClassWrapped.__wrapped__, "__wrapped__")
        assert ClassWrapped.__wrapped__.__wrapped__ == Class

        a = Class()
        b = ClassWrapped()

        assert hasattr(a, "__special_object__")
        assert hasattr(b, "__special_object__")
        assert b.__special_object__ == a.__special_object__


    def test_class_static_variables(self):
        class Class(object):
            pass

        ClassWrapped = nanshe.util.wrappers.class_static_variables(a=5)(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class

        assert not hasattr(Class, "a")
        assert hasattr(ClassWrapped, "a")
        assert ClassWrapped.a == 5


    def test_class_decorate_all_methods(self):
        class Class(object):
            def __init__(self):
                pass

        ClassWrapped = nanshe.util.wrappers.class_decorate_all_methods(
            nanshe.util.wrappers.identity_wrapper
        )(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class

        assert ClassWrapped.__init__ != Class.__init__
        assert not hasattr(Class.__init__, "__wrapped__")
        assert hasattr(ClassWrapped.__init__, "__wrapped__")

        if sys.version_info.major < 3:
            assert ClassWrapped.__init__.__wrapped__ != Class.__init__
        else:
            assert ClassWrapped.__init__.__wrapped__ == Class.__init__

        assert ClassWrapped.__wrapped__.__init__ == Class.__init__


    def test_qt_class_decorate_all_methods(self):
        class Class(PyQt4.QtCore.QObject):
            def __init__(self):
                pass

        ClassWrapped = nanshe.util.wrappers.qt_class_decorate_all_methods(
            nanshe.util.wrappers.identity_wrapper
        )(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class

        assert ClassWrapped.__init__ != Class.__init__
        assert not hasattr(Class.__init__, "__wrapped__")
        assert hasattr(ClassWrapped.__init__, "__wrapped__")

        if sys.version_info.major < 3:
            assert ClassWrapped.__init__.__wrapped__ != Class.__init__
        else:
            assert ClassWrapped.__init__.__wrapped__ == Class.__init__

        assert ClassWrapped.__wrapped__.__init__ == Class.__init__


    def test_class_decorate_methods(self):
        class Class(object):
            def __init__(self):
                pass

            def func_0(self):
                pass

        ClassWrapped = nanshe.util.wrappers.class_decorate_methods(
            func_0=nanshe.util.wrappers.identity_wrapper
        )(Class)

        assert ClassWrapped != Class
        assert not hasattr(Class, "__wrapped__")
        assert hasattr(ClassWrapped, "__wrapped__")
        assert ClassWrapped.__wrapped__ == Class

        assert ClassWrapped.__init__ == Class.__init__
        assert not hasattr(Class.__init__, "__wrapped__")
        assert not hasattr(ClassWrapped.__init__, "__wrapped__")
        assert ClassWrapped.__wrapped__.__init__ == Class.__init__

        assert ClassWrapped.func_0 != Class.func_0
        assert not hasattr(Class.func_0, "__wrapped__")
        assert hasattr(ClassWrapped.func_0, "__wrapped__")

        if sys.version_info.major < 3:
            assert ClassWrapped.func_0.__wrapped__ != Class.func_0
        else:
            assert ClassWrapped.func_0.__wrapped__ == Class.func_0

        assert ClassWrapped.__wrapped__.func_0 == Class.func_0

    def test_unwrap(self):
        def func_0():
            pass

        func_1 = nanshe.util.wrappers.identity_wrapper(func_0)
        func_2 = nanshe.util.wrappers.identity_wrapper(func_1)

        assert nanshe.util.wrappers.unwrap(func_1) == func_0
        assert nanshe.util.wrappers.unwrap(func_2) != func_1
        assert nanshe.util.wrappers.unwrap(func_2) == func_0

    def test_tied_call_args(self):
        def func_0(a, b=5, *v, **k):
            return(a + b + sum(v) + sum(list(k.values())))

        tied_args, args, kwargs = nanshe.util.wrappers.tied_call_args(
            func_0, 1
        )
        assert list(tied_args.items()) == [("a", 1), ("b", 5)]
        assert args == tuple()
        assert list(kwargs.items()) == []

        tied_args, args, kwargs = nanshe.util.wrappers.tied_call_args(
            func_0, a=1, c=7
        )
        assert list(tied_args.items()) == [("a", 1), ("b", 5)]
        assert args == tuple()
        assert list(kwargs.items()) == [("c", 7)]

        tied_args, args, kwargs = nanshe.util.wrappers.tied_call_args(
            func_0, 1, 2, 3, c=7
        )
        assert list(tied_args.items()) == [("a", 1), ("b", 2)]
        assert args == (3,)
        assert list(kwargs.items()) == [("c", 7)]

    def test_repack_call_args(self):
        def func_0(a, b=5, *v, **k):
            return(a + b + sum(v) + sum(list(k.values())))

        args, kwargs = nanshe.util.wrappers.repack_call_args(func_0, 1)
        assert args == (1,)
        assert list(kwargs.items()) == [("b", 5)]

        args, kwargs = nanshe.util.wrappers.repack_call_args(
            func_0, a=1, c=7
        )

        assert args == tuple()
        assert sorted(kwargs.items()) == [("a", 1), ("b", 5), ("c", 7)]

        args, kwargs = nanshe.util.wrappers.repack_call_args(
            func_0, 1, 2, 3, c=7
        )
        assert args == (1, 2, 3)
        assert list(kwargs.items()) == [("c", 7)]


def setup_with_setup_state_2(a_callable):
    print("setup_2")
    assert not hasattr(a_callable, "a")


def setup_with_setup_state_1(a_callable):
    print("setup_1")
    setattr(a_callable, "a", 5)


def teardown_with_setup_state_1(a_callable):
    print("teardown_1")
    delattr(a_callable, "a")


def teardown_with_setup_state_2(a_callable):
    print("teardown_2")
    assert not hasattr(a_callable, "a")


@nanshe.util.wrappers.with_setup_state()
def test_with_setup_state_1a():
    print("test")


@nanshe.util.wrappers.with_setup_state(setup_with_setup_state_1,
                                       teardown_with_setup_state_1)
def test_with_setup_state_1b():
    print("test")
    assert hasattr(test_with_setup_state_1b, "a")
    assert getattr(test_with_setup_state_1b, "a") == 5


@nanshe.util.wrappers.with_setup_state(setup_with_setup_state_2,
                                       teardown_with_setup_state_2)
@nanshe.util.wrappers.with_setup_state(setup_with_setup_state_1,
                                       teardown_with_setup_state_1)
def test_with_setup_state_2a():
    print("test")
    assert hasattr(test_with_setup_state_2a, "a")
    assert getattr(test_with_setup_state_2a, "a") == 5


@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state(setup_with_setup_state_2,
                                       teardown_with_setup_state_2)
def test_with_setup_state_2b():
    print("test")
    assert not hasattr(test_with_setup_state_2b, "a")


@nanshe.util.wrappers.with_setup_state(setup_with_setup_state_2,
                                       teardown_with_setup_state_2)
@nanshe.util.wrappers.with_setup_state()
def test_with_setup_state_2c():
    print("test")
    assert not hasattr(test_with_setup_state_2c, "a")


@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
def test_with_setup_state_2d():
    print("test")
    assert not hasattr(test_with_setup_state_2c, "a")


@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
@nanshe.util.wrappers.with_setup_state()
def test_with_setup_state_6():
    print("test")
    assert not hasattr(test_with_setup_state_6, "a")
