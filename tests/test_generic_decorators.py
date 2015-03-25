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

        func = nanshe.nanshe.generic_decorators.static_variables(
            c = 7
        )(
            func
        )

        assert hasattr(func, "c")
        assert func.c == 7
