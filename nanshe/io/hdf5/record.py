"""
The module ``record`` provides the equivalent of a logger for array data.

===============================================================================
Overview
===============================================================================
Applying the decorator ``static_subgrouping_array_recorders`` with the names of
desired recorders as strings or keyword arguments with a given recorder
instance initializes the ``recorders`` attribute, which contains each recorder
as a dictionary would. Results are stored under a group named after decorated
function. Nested function calls can be paralleled by nesting results within
groups. All that is required is that the recorder be assigned from outer scope
to inner (see example below).

.. code:: python

    @static_subgrouping_array_recorders("r")
    def a(x):
        a.r = x
        return x * x.T

    @static_subgrouping_array_recorders("r")
    def b(x, y):
        a.r = b.r
        return a(x + y)

All recorders default to an instance of ``EmptyArrayRecorder`` unless otherwise
specified. All operations work on them exactly the same, but no recording
occurs. This guarantees that code using recorders will not break if they don't
have them. A similar strategy can be applied for classes and their methods by
using ``class_static_array_debug_recorder``.

.. todo:: Complete documentation with more examples.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 04, 2014 11:10:55 EDT$"


import numpy
import h5py

import serializers
from nanshe.util import wrappers


# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_meta_logger = prof.getTraceMetaLogger(__name__)


@prof.log_class(trace_meta_logger)
class EmptyArrayRecorder(object):
    def __init__(self):
        self.__recorders = set()

    def __nonzero__(self):
        return(False)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    def get(self, key, default=None):
        value = default

        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return(value)

    def __contains__(self, key):
        return(key in self.__recorders)

    def __getitem__(self, key):
        if (key == "."):
            return(self)

        if key in self.__recorders:
            return(EmptyArrayRecorder())
        else:
            raise(KeyError(
                "unable to open object (Symbol table: Can't open object " +
                repr(key) + ")"
            ))

    def __setitem__(self, key, value):
        # Exception will be thrown if value is empty or if key already exists
        # (as intended).
        if (key == "."):
            if not ((value is None) or (value is h5py.Group)):
                raise ValueError("Cannot store dataset in top level group.")
        elif (value is None) or (value is h5py.Group):
            self.__recorders.add(key)
        else:
            if value.size:
                pass
            else:
                raise ValueError(
                    "The array provided for output by the name: \"" + key +
                    "\" is empty."
                )


@prof.log_class(trace_meta_logger)
class HDF5ArrayRecorder(object):
    def __init__(self, hdf5_handle, overwrite=False):
        self.hdf5_handle = hdf5_handle
        self.overwrite = overwrite

    def __nonzero__(self):
        return(True)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    def get(self, key, default=None):
        value = default

        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return(value)

    def __contains__(self, key):
        return(key in self.hdf5_handle)

    def __getitem__(self, key):
        if (key == "."):
            return(self)

        try:
            if isinstance(self.hdf5_handle[key], h5py.Group):
                return(HDF5ArrayRecorder(
                    self.hdf5_handle[key], overwrite=self.overwrite
                ))
            else:
                return(serializers.read_numpy_structured_array_from_HDF5(
                    self.hdf5_handle, key
                ))
        except:
            raise(KeyError(
                "unable to open object (Symbol table: Can't open object " +
                repr(key) + " in " + repr(self.hdf5_handle) + ")"
            ))

    def __setitem__(self, key, value):
        if (key == "."):
            if not ((value is None) or (value is h5py.Group)):
                raise ValueError(
                    "Cannot store dataset in top level group ( " +
                    self.hdf5_handle.name + " )."
                )

            if self.overwrite:
                for each_key in self.hdf5_handle:
                    del self.hdf5_handle[each_key]

                self.hdf5_handle.file.flush()
        elif (value is None) or (value is h5py.Group):
            # Check to see if the output must go somewhere special.
            if key:
                # If so, check to see if it exists.
                if key in self.hdf5_handle:
                    # If it does and we want to overwrite it, do so.
                    if self.overwrite:
                        del self.hdf5_handle[key]

                        self.hdf5_handle.create_group(key)

                        self.hdf5_handle.file.flush()
                else:
                    # Create it if it doesn't, exist.
                    self.hdf5_handle.create_group(key)

                    self.hdf5_handle.file.flush()
        else:
            # Attempt to create a dataset in self.hdf5_handle named key with
            # value and do not overwrite. Exception will be thrown if value is
            # empty or if key already exists (as intended).
            if value.size:
                serializers.create_numpy_structured_array_in_HDF5(
                    self.hdf5_handle,
                    key,
                    value,
                    overwrite=self.overwrite
                )
                self.hdf5_handle.file.flush()

                return()
            else:
                raise ValueError(
                    "The array provided for output by the name: \"" +
                    key + "\" is empty."
                )


@prof.log_class(trace_meta_logger)
class HDF5EnumeratedArrayRecorder(object):
    def __init__(self, hdf5_handle):
        self.hdf5_handle = hdf5_handle

        # Must be a logger if it already exists.
        assert self.hdf5_handle.attrs.get("is_logger", True)

        self.hdf5_handle.attrs["is_logger"] = True
        self.hdf5_handle.file.flush()

        self.hdf5_index_data_handles = {"." : -1}
        for each_index in self.hdf5_handle:
            each_index = int(each_index)
            self.hdf5_index_data_handles["."] = max(
                self.hdf5_index_data_handles["."], each_index
            )

        if self.hdf5_index_data_handles["."] != -1:
            hdf5_index_handle = self.hdf5_handle[str(
                self.hdf5_index_data_handles["."]
            )]
            for each_key in hdf5_index_handle:
                if hdf5_index_handle[each_key].attrs.get("is_logger", False):
                    self.hdf5_index_data_handles[each_key] = None
                else:
                    self.hdf5_index_data_handles[each_key] = -1

                    for each_index in hdf5_index_handle[each_key]:
                        each_index = int(each_index)
                        self.hdf5_index_data_handles[each_key] = max(
                            self.hdf5_index_data_handles[each_key], each_index
                        )

    def __nonzero__(self):
        return(True)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    def get(self, key, default=None):
        value = default

        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return(value)

    def __contains__(self, key):
        return(self.get(key) is not None)

    def __getitem__(self, key):
        if (key == "."):
            return(self)
        try:
            root_i = self.hdf5_index_data_handles.get(".", -1)
            key_i = self.hdf5_index_data_handles.get(key, -1)

            assert (root_i != -1)
            assert (key_i != -1)

            root_i_str = str(root_i)

            assert isinstance(self.hdf5_handle[root_i_str], h5py.Group)
            assert isinstance(self.hdf5_handle[root_i_str][key], h5py.Group)

            key_handle = self.hdf5_handle[root_i_str][key]
            if key_i is None:
                return(HDF5EnumeratedArrayRecorder(key_handle))
            else:
                key_i_str = str(key_i)
                return(serializers.read_numpy_structured_array_from_HDF5(
                    key_handle, key_i_str
                ))
        except:
            raise(KeyError(
                "unable to open object (Symbol table: Can't open object " +
                repr(key) + " in " + repr(self.hdf5_handle) + ")"
            ))

    def __setitem__(self, key, value):
        if (key == "."):
            if not ((value is None) or (value is h5py.Group)):
                raise ValueError(
                    "Cannot store dataset in top level group ( " +
                    self.hdf5_handle.name + " )."
                )

            self.hdf5_index_data_handles = {
                "." : self.hdf5_index_data_handles["."] + 1
            }
            self.hdf5_handle.create_group(
                str(self.hdf5_index_data_handles["."])
            )
            self.hdf5_handle.file.flush()
        else:
            hdf5_index_handle = None
            try:
                hdf5_index_handle = self.hdf5_handle[str(
                    self.hdf5_index_data_handles["."]
                )]
            except KeyError:
                if self.hdf5_index_data_handles["."] == -1:
                    self.hdf5_index_data_handles = {
                        "." : self.hdf5_index_data_handles["."] + 1
                    }

                self.hdf5_handle.create_group(
                    str(self.hdf5_index_data_handles["."])
                )
                self.hdf5_handle.file.flush()

                hdf5_index_handle = self.hdf5_handle[str(
                    self.hdf5_index_data_handles["."]
                )]

            if (value is None) or (value is h5py.Group):
                # Create a group if it doesn't already exist.
                hdf5_index_handle.require_group(key)
                hdf5_index_handle.attrs["is_logger"] = True
                hdf5_index_handle.file.flush()
                self.hdf5_index_data_handles[key] = None
            else:
                # Index into a NumPy structured array can return a void type
                # even though it is a valid array, which can be stored.
                # So, we must check.
                try:
                    assert isinstance(value, numpy.ndarray)
                except AssertionError:
                    if not value.dtype.names:
                        raise
                if value.size:
                    # If so, check to see if it exists.
                    if key not in self.hdf5_index_data_handles:
                        hdf5_index_handle.create_group(key)
                        hdf5_index_handle[key].attrs["is_logger"] = False
                        hdf5_index_handle.file.flush()
                        self.hdf5_index_data_handles[key] = -1

                    self.hdf5_index_data_handles[key] += 1

                    serializers.create_numpy_structured_array_in_HDF5(
                        hdf5_index_handle[key],
                        str(self.hdf5_index_data_handles[key]),
                        value
                    )

                    self.hdf5_handle.file.flush()
                else:
                    raise ValueError(
                        "The array provided for output by the name: \"" +
                        key + "\" is empty."
                    )


@prof.log_call(trace_meta_logger)
def generate_HDF5_array_recorder(hdf5_handle,
                                 group_name="",
                                 enable=True,
                                 overwrite_group=False,
                                 recorder_constructor=HDF5ArrayRecorder,
                                 **kwargs):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group in an HDF5 file.

        Args:
            hdf5_handle:            The HDF5 file group to place the debug
                                    contents into.

            group_name:             The name of the group within hdf5_handle to
                                    save the contents to. (If set to the empty
                                    string, data will be saved to
                                    hdf5_handle directly)

            enable:                 Whether to generate a real recorder or a
                                    fake one.

            overwrite_group:        Whether to overwrite the group where data
                                    is stored.

            recorder_constructor:   Type of recorder to use if enable is True.

            **kwargs:               Other arguments to pass through to the
                                    recorder_constructor (won't pass through if
                                    enable is false).

        Returns:
            ArrayRecorder:          A function, which will take a given array
                                    name and value and write them out.
    """

    if isinstance(hdf5_handle, str):
        hdf5_handle = h5py.File(hdf5_handle, "a")

    if (enable):
        hdf5_recording_handle = hdf5_handle

        # Check to if the output must go somewhere special.
        if group_name:
            # If so, check to see if it exists.
            if group_name in hdf5_handle:
                # If it does and we want to overwrite it, do so.
                if overwrite_group:
                    del hdf5_handle[group_name]

                    hdf5_handle.create_group(group_name)

                    hdf5_handle.file.flush()
            else:
                # Create it if it doesn't, exist.
                hdf5_handle.create_group(group_name)

                hdf5_handle.file.flush()

            hdf5_recording_handle = hdf5_handle[group_name]

        return(recorder_constructor(hdf5_recording_handle, **kwargs))
    else:
        return(EmptyArrayRecorder())


@prof.log_call(trace_meta_logger)
def static_subgrouping_array_recorders(*args, **kwargs):
    """
        Creates a decorator that adds a static variable, recorders, that holds
        as many recorders as are supplied.

        Args:
            args(tuple of strs):                        All variables to be
                                                        named (set to
                                                        EmptyArrayRecorder()).

        Keyword Args:
            kwargs(dict of strs and ArrayRecorders):    All variables to be
                                                        named with values of
                                                        type ArrayRecorder.

        Returns:
            (callable):                                 A decorator that adds
                                                        the static variable,
                                                        recorders, to the given
                                                        function.
    """

    @prof.log_call(trace_meta_logger)
    def static_subgrouping_array_recorders_tie(callable):
        """
            Creates a decorator that adds a static variable recorders to the
            function it decorates.

            Args:
                callable(callable):     All variables to be named (set to
                                        EmptyArrayRecorder()).

            Returns:
                (callable):             A function with the static variable,
                                        recorders, added.
        """

        class SubgroupingRecorders(object):
            # """
            #     Hold recorders. Automatically, moves instances of
            #     ArrayRecorder to a subgroup with the same name as the
            #     callable on assignment.
            # """
            def __init__(self, *args, **kwargs):
                # """
                #     Contains ArrayRecorders that write to a subgroup of the
                #     same name as the callable.
                #
                #     Args:
                #         args(tuple of strs):                        All
                #                                                     variables
                #                                                     to be
                #                                                     named
                #                                                     (set to
                #                                                     EmptyArrayRecorder()).
                #
                #     Keyword Args:
                #         kwargs(dict of strs and ArrayRecorders):    All
                #                                                     variables
                #                                                     to be
                #                                                     named
                #                                                     with
                #                                                     values of
                #                                                     type
                #                                                     ArrayRecorder.
                # """

                for _k in args:
                    object.__setattr__(self, _k, None)

                for _k, _v in kwargs.items():
                    object.__setattr__(self, _k, _v)

            def __getattr__(self, _k):
                if _k != "__dict__":
                    return(self.__dict__[_k])

            def __setattr__(self, _k, _v):
                if _k != "__dict__":
                    if _v is None:
                        self.__dict__[_k] = EmptyArrayRecorder()
                    else:
                        _v[callable.__name__] = None
                        try:
                            self.__dict__[_k] = _v[callable.__name__]
                        except KeyError:
                            if isinstance(_v, EmptyArrayRecorder):
                                self.__dict__[_k] = EmptyArrayRecorder()
                            else:
                                raise

            def __delattr__(self, _k):
                if _k != "__dict__":
                    del self.__dict__[_k]

        callable = wrappers.static_variables(
            recorders=SubgroupingRecorders(*args, **kwargs)
        )(callable)

        @wrappers.wraps(callable)
        def static_subgrouping_array_recorders_wrapper(*args, **kwargs):
            # Force all recorders to ensure their output Group exists.
            # All of them actually make the directory.
            # However, HDF5EnumeratedArrayRecorder needs a clue as to
            # when it should switch to a new one as it will keep different
            # runs separate.
            for _k in callable.recorders.__dict__:
                callable.recorders.__dict__[_k]["."] = None

            return(callable(*args, **kwargs))

        return(static_subgrouping_array_recorders_wrapper)

    return(static_subgrouping_array_recorders_tie)


@prof.log_call(trace_meta_logger)
def static_array_debug_recorder(callable):
    """
        Creates a decorator that adds a static variable recorders that contains
        the variable array_debug_recorder to the function it decorates.
        By
        default, array_debug_recorder is set to an EmptyArrayRecorder instance.
        Also, on assignment it automatically creates a subgroup with the same
        name as the function.

        Args:
            callable(callable):    All variables to be named (set to
                                   EmptyArrayRecorder()).

        Returns:
            (callable):            A decorator that adds the static variable
                                   array_debug_recorder to the given function.
    """

    callable = static_subgrouping_array_recorders(
        array_debug_recorder=EmptyArrayRecorder()
    )(callable)

    return(callable)


@prof.log_call(trace_meta_logger)
def class_static_subgrouping_array_recorders(*args, **kwargs):
    """
        Creates a decorator that adds a static variable, recorders, that holds
        as many recorders as are supplied.

        Args:
            args(tuple of strs):                        All variables to be
                                                        named (set to
                                                        EmptyArrayRecorder()).

        Keyword Args:
            kwargs(dict of strs and ArrayRecorders):    All variables to be
                                                        named with values of
                                                        type ArrayRecorder.

        Returns:
            (callable):                                 A decorator that adds
                                                        the static variable,
                                                        recorders, to the given
                                                        function.
    """

    @prof.log_call(trace_meta_logger)
    def class_static_subgrouping_array_recorders_tie(a_class):
        """
            Creates a decorator that adds a static variable recorders to the
            function it decorates.

            Args:
                a_class(class):      All variables to be named (set to
                                     EmptyArrayRecorder()).

            Returns:
                (class):             A function with the static variable,
                                     recorders, added.
        """

        class ClassSubgroupingRecorders(object):
            # """
            #     Hold recorders. Automatically, moves instances of
            #     ArrayRecorder to a subgroup with the same name as the a_class
            #     on assignment.
            # """
            def __init__(self, *args, **kwargs):
                # """
                #     Contains ArrayRecorders that write to a subgroup of the
                #     same name as the a_class.
                #
                #     Args:
                #         args(tuple of strs):                       All
                #                                                    variables
                #                                                    to be named
                #                                                    (set to
                #                                                    EmptyArrayRecorder()).
                #
                #     Keyword Args:
                #         kwargs(dict of strs and ArrayRecorders):   All
                #                                                    variables
                #                                                    to be named
                #                                                    with values
                #                                                    of type
                #                                                    ArrayRecorder.
                # """

                for _k in args:
                    object.__setattr__(self, _k, None)

                for _k, _v in kwargs.items():
                    object.__setattr__(self, _k, _v)

            def __getattr__(self, _k):
                if _k != "__dict__":
                    return(self.__dict__[_k])

            def __setattr__(self, _k, _v):
                if _k != "__dict__":
                    if _v is None:
                        self.__dict__[_k] = EmptyArrayRecorder()
                    else:
                        _v[a_class.__name__] = None
                        try:
                            self.__dict__[_k] = _v[a_class.__name__]
                        except KeyError:
                            if isinstance(_v, EmptyArrayRecorder):
                                self.__dict__[_k] = EmptyArrayRecorder()
                            else:
                                raise

            def __delattr__(self, _k):
                if _k != "__dict__":
                    del self.__dict__[_k]

        a_class = wrappers.class_static_variables(
            recorders=ClassSubgroupingRecorders(*args, **kwargs)
        )(a_class)

        def class_static_subgrouping_array_recorders__init__decorator(callable):
            @wrappers.wraps(callable)
            def class_static_subgrouping_array_recorders__init__wrapper(self,
                                                                        *args,
                                                                        **kwargs):
                # Force all recorders to ensure their output Group exists.
                # All of them actually make the directory.
                # However, HDF5EnumeratedArrayRecorder needs a clue as to
                # when it should switch to a new one as it will keep different
                # runs separate.

                self.recorders = ClassSubgroupingRecorders()
                for _k in a_class.recorders.__dict__:
                    self.recorders.__dict__[_k] = a_class.recorders.__dict__[_k]

                return(callable(self, *args, **kwargs))

            return(class_static_subgrouping_array_recorders__init__wrapper)

        def class_static_subgrouping_array_recorders_decorator(callable):
            callable = static_subgrouping_array_recorders()(callable)

            @wrappers.wraps(callable)
            def class_static_subgrouping_array_recorders_wrapper(self,
                                                                 *args,
                                                                 **kwargs):
                # Force all recorders to ensure their output Group exists.
                # All of them actually make the directory.
                # However, HDF5EnumeratedArrayRecorder needs a clue as to
                # when it should switch to a new one as it will keep different
                # runs separate.
                callable.recorders.__dict__ = dict()
                for _k in self.recorders.__dict__:
                    # setattr(callable.recorders, _k,
                    # self.recorders.__dict__[_k][callable.__name__)])
                    setattr(
                        callable.recorders, _k, getattr(self.recorders, _k)
                    )
                    # callable.recorders.__dict__[_k]["."] = None

                return(callable(self, *args, **kwargs))

            return(class_static_subgrouping_array_recorders_wrapper)

        # Wraps __init__ only
        a_class = wrappers.class_decorate_methods(
            __init__=class_static_subgrouping_array_recorders__init__decorator
        )(a_class)

        # Wrap everything
        a_class = wrappers.class_decorate_all_methods(
            class_static_subgrouping_array_recorders_decorator)(a_class)


        # Must be done last.
        # Precedes the constructor to ensure a new working directory is created
        class MetaSubgroupingRecorders(type):
            def __call__(self, *args, **kwargs):
                for _k in self.recorders.__dict__:
                    self.recorders.__dict__[_k]["."] = None

                return(super(MetaSubgroupingRecorders, self).__call__(
                    *args, **kwargs
                ))

        a_class = wrappers.metaclass(MetaSubgroupingRecorders)(a_class)

        return(a_class)

    return(class_static_subgrouping_array_recorders_tie)


@prof.log_call(trace_meta_logger)
def class_static_array_debug_recorder(a_class):
    """
        Creates a decorator that adds a static variable recorders that contains
        the variable array_debug_recorder to the function it decorates. By
        default, array_debug_recorder is set to an EmptyArrayRecorder instance.
        Also, on assignment it automatically creates a subgroup with the same
        name as the function.

        Args:
            a_class(class):    All variables to be named
                               (set to EmptyArrayRecorder()).

        Returns:
            (class):           A decorator that adds the static variable
                               array_debug_recorder to the given function.
    """

    a_class = class_static_subgrouping_array_recorders(
        array_debug_recorder=EmptyArrayRecorder()
    )(a_class)

    return(a_class)
