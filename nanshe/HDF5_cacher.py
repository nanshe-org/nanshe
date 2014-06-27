""" As some elements of our pipeline take a fair amount of time to generate, it makes sense to have some sort of caching
    mechanism. Namely, one that uses the same HDF5 file that we already use. Ideally, the only things that will be cached
    are things that take a long time to determine. Short calls can be recalculated easily and so the price of time to
    calculate is small compared to the space requirements of an image or the time spent finding it. Long calls like
    dictionary learning, finding neurons, etc. take significantly more time than the cost of caching them or even
    searching for them in the cache. The simplest implementation should act as a callback ."""

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 11, 2014 14:28:25 EDT$"



import sys
import json
import hashlib
import multiprocessing

import h5py
import numpy

import pathHelpers


import HDF5_serializers

# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)

cache_lock = multiprocessing.RLock()
cache_group_global = None
cache_group_local = None


def startup():
    global cache_lock
    global cache_group_global
    global cache_group_local

    with cache_lock:
        if cache_group_global is None:
            cache_group_global = h5py.File("tmp.h5", driver = "core", backing_store = False)
        elif isinstance(cache_group_global, str):
            cache_group_path_components = pathHelpers.PathComponents(cache_group_global)

            cache_group_global = h5py.File(cache_group_path_components.externalPath, "a")

            cache_group_global = cache_group_global[cache_group_path_components.internalPath]
        elif isinstance(cache_group_global, h5py.Group):
            pass

        cache_group_local = cache_group_global


def get_location_relative_global_cache(new_loc):
    if isinstance(new_loc, h5py.Group) or isinstance(new_loc, h5py.Dataset):
        new_loc = new_loc.name

    if isinstance(new_loc, str):
        return(new_loc.replace(cache_group_global.name, "", 1))
    else:
        raise Exception("Unknown type for new_loc argument as " + repr(type(new_loc)))


class ndarrayCached(numpy.ndarray):
    """
        Extends the numpy.ndarray to keep links to input used and output dependent on the results.

        This borrows suggestions from ( http://docs.scipy.org/doc/numpy/user/basics.subclassing.html ).
    """

    def __new__(cls, *args, **kwargs):
        """
            Called before __init__ to construct the object. Used by numpy.ndarray. So, this determines what numpy.ndarray
            will get.

            Arguments:
                cls:            could be a type or the instance to construct.
                *args:          arguments to forward
                **kwargs:       keyword arguments to forward.
        """

        obj = None

        if type(args[0]) is tuple:
            def using_raw_ndarray_constructor(subtype,
                                              shape,
                                              dtype=float,
                                              buffer=None,
                                              offset=0,
                                              strides=None,
                                              order=None,
                                              prev_link="",
                                              cur_link = "",
                                              next_links=[]):
                obj = numpy.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)

                obj.prev_link = prev_link
                obj.cur_link = cur_link
                obj.next_links = next_links

                return(obj)

            obj = using_raw_ndarray_constructor(cls, *args, **kwargs)
        elif isinstance(args[0], h5py.Dataset):
            def using_hdf5_dataset_constructor(dataset):
                obj = numpy.asarray(HDF5_serializers.read_numpy_structured_array_from_HDF5(dataset.file, dataset.name)).view(cls)

                obj.prev_link = str()
                obj.cur_link = dataset.name
                obj.next_links = list()

                return(obj)

            obj = using_hdf5_dataset_constructor(cls, *args, **kwargs)
        elif isinstance(args[0], h5py.Group) or isinstance(args[0], unicode) or isinstance(args[0], str):
            def using_hdf5_group_constructor(group):
                obj = numpy.asarray(HDF5_serializers.read_numpy_structured_array_from_HDF5(group, "output_array")).view(cls)

                obj.prev_link = str(group.attrs["prev_link"])
                obj.cur_link = group.name
                obj.next_links = list(group.attrs["next_links"])

                return(obj)

            obj = using_hdf5_group_constructor(cls, *args, **kwargs)
        elif isinstance(args[0], numpy.ndarray):
            if isinstance(args[0], ndarrayCached):
                def using_cached_array_constructor(cls, an_array_cached):
                    obj = numpy.asarray(an_array_cached).view(cls)

                    obj.prev_link = an_array_cached.prev_link
                    obj.cur_link = an_array_cached.cur_link
                    obj.next_links = an_array_cached.next_links

                    return(obj)

                obj = using_cached_array_constructor(cls, *args, **kwargs)
            else:
                def using_array_ndarray_constructor(cls, an_array, prev_link="", cur_link = "", next_links=[]):
                    obj = numpy.asarray(an_array).view(cls)

                    obj.prev_link = prev_link
                    obj.cur_link = cur_link
                    obj.next_links = next_links

                    return(obj)

                obj = using_array_ndarray_constructor(cls, *args, **kwargs)
        else:
            raise Exception("Unknown arguments.")

        return(obj)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.prev_link = getattr(obj, "prev_link", "")
            self.cur_link = getattr(obj, "cur_link", "")
            self.next_links = getattr(obj, "next_links", [])

    def store_cache(self):
        if self.cur_link not in cache_group_global:
            cache_group_global.create_group(self.cur_link)

        self_cache = cache_group_global[self.cur_link]

        if "prev_link" in self_cache.attrs:
            del self_cache.attrs["prev_link"]

        self_cache["prev_link"] = self.prev_link

        if "next_links" in self_cache.attrs:
            del self_cache.attrs["next_links"]

        self_cache["next_links"] = self.next_links

        if "input_array" not in self_cache:
            self_cache["input_array"] = cache_group_global[self.prev_link]

        if "output_array" not in self_cache:
            HDF5_serializers.write_numpy_structured_array_to_HDF5(self_cache, "output_array", self, overwrite = True)



class HDF5Cache(object):
    def __init__(self, a_callable):
        self.__callable = a_callable
        self.__callable_name = ""

        callable_module = sys.modules[self.__callable.__module__]

        self.__callable_name += getattr(callable_module, "__file__", callable_module.__name__)
        self.__callable_name += "."
        self.__callable_name += self.__callable.__name__

    def __call__(self, input_array, *args, **kwargs):
        global cache_lock
        global cache_group_local
        global cache_group_local

        with cache_lock:
            output_array = None

            if not isinstance(input_array, ndarrayCached):
                input_array = ndarrayCached(input_array)

            parameters = { self.__callable_name : {
                                                    "input_array" : input_array.cur_link,
                                                    "args" : list(args),
                                                    "kwargs" : dict(kwargs)
                         }
            }
            parameters_str = json.dumps(parameters)
            parameter_hash = hashlib.sha1(parameters_str).digest()

            if parameter_hash in input_array.next_links:
                # found result!
                output_array = ndarrayCached(cache_group_local[parameter_hash])
            elif parameter_hash in cache_group_local:
                # found result!
                # not linked though

                output_array = ndarrayCached(cache_group_local[parameter_hash])

                input_array.next_links.append(cache_group_local[parameter_hash].name)

                output_array.prev_link = input_array.cur_link

                input_array.store()

                output_array.store()
            else:
                # ugh, work.

                # create dir for function params, cache, results
                cache_group_local.create_group(parameter_hash)
                cache_group_local = cache_group_local[parameter_hash]

                # store params as json str
                cache_group_local["parameters"] = numpy.array(parameters_str)

                # create cache for nested function calls (if needed)
                cache_group_local.create_group("cache")
                cache_group_local = cache_group_local["cache"]

                # if the input array is a str link based on the the cache group, pull it from the cache.
                if input_array is str:
                    input_array = ndarrayCached(cache_group_global[input_array])

                # call for result
                output_array = self.__callable(input_array, *args, **kwargs)

                # get out of the function cache back into the function dir
                cache_group_local = cache_group_local.parent

                # convert the result into a new ndarrayCached that references the previous result
                output_array = ndarrayCached(output_array.view(numpy.ndarray),
                                             prev_link = input_array.cur_link,
                                             cur_link = cache_group_local.name,
                                             next_links = [])

                # get out of the function dir (now in some other function's or global's cache dir)
                cache_group_local = cache_group_local.parent

                # Set links between input_array and result



            return(output_array)