__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 25, 2015 13:30:52 EDT$"


import functools

import nanshe.nanshe.generic_decorators


class TestGenericDecorators(object):
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

        func_wrapped_2 = nanshe.nanshe.generic_decorators.update_wrapper(
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

        func_wrapped_2 = nanshe.nanshe.generic_decorators.wraps(wrapper)(
            func
        )

        assert func_wrapped_1 == func_wrapped_2


    def test_identity_wrapper(self):
        def func(a, b=2):
            return(a + b)

        func_wrapped = nanshe.nanshe.generic_decorators.identity_wrapper(
            func
        )

        assert func_wrapped != func
        assert not hasattr(func, "__wrapped__")
        assert hasattr(func_wrapped, "__wrapped__")
        assert func_wrapped.__wrapped__ == func


    def test_static_variables(self):
        def func(a, b=2):
            return(a + b)

        func_wrapped = nanshe.nanshe.generic_decorators.static_variables(
            c = 7
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

        ClassWrapped = nanshe.nanshe.generic_decorators.metaclass(Meta)(Class)

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

        ClassWrapped = nanshe.nanshe.generic_decorators.metaclass(Meta)(Class)

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

        ClassWrapped = nanshe.nanshe.generic_decorators.metaclasses(
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

        ClassWrapped = nanshe.nanshe.generic_decorators.metaclasses(
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
