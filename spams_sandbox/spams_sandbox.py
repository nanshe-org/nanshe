__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 20, 2014 12:07:48 EDT$"


#import nanshe.advanced_debugging


#logger = nanshe.advanced_debugging.logging.getLogger(__name__)




class SPAMSException(Exception):
    pass


#@nanshe.advanced_debugging.log_call(logger)
def run_multiprocessing_queue_spams_trainDL(out_queue, *args, **kwargs):
    """
        Designed to run spams.trainDL in a separate process.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Queue to return the resulting dictionary.


        Args:
            out_queue(multiprocessing.Queue):       what will take the returned dictionary from spams.trainDL.
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            Todo
            Look into having the raw data for input for spams.trainDL copied in.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    out_queue.put(result)


#@nanshe.advanced_debugging.log_call(logger)
def call_multiprocessing_queue_spams_trainDL(*args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Queue to return the resulting dictionary.


        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            Todo
            Look into having the raw data for input for spams.trainDL copied in.

        Returns:
            result(numpy.matrix): the dictionary found
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    out_queue = multiprocessing.Queue()

    p = multiprocessing.Process(target = run_multiprocessing_queue_spams_trainDL, args = (out_queue,) + args, kwargs = kwargs)
    p.start()
    result = out_queue.get()
    result = result.copy()
    p.join()

    if p.exitcode != 0:
        raise SPAMSException("SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\".")

    return(result)


#@nanshe.advanced_debugging.log_call(logger)
def run_multiprocessing_array_spams_trainDL(result_array, X, *args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.


        Args:
            result_array(multiprocessing.Array):    shared memory array to store results in.
            X(numpy.ndarray):                       currently uses numpy ndarray as input.
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.

            Todo
            Need to deal with return_model case.
            Look into having the raw data for input for spams.trainDL copied in.
    """

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    # Also, it is not needed outside of calling this function.
    import spams

    # Create a numpy.ndarray that uses the shared buffer.
    result = numpy.frombuffer(result_array.get_obj(), dtype = X.dtype).reshape((-1, kwargs["K"]))

    result[:] = spams.trainDL(X, *args, **kwargs)


#@nanshe.advanced_debugging.log_call(logger)
def call_multiprocessing_array_spams_trainDL(X, *args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.


        Args:
            X(numpy.matrix):                        a Fortran order NumPy Matrix with the same name as used by spams.trainDL (so if someone tries to use it as a keyword argument...).
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.

            Todo
            Need to deal with return_model case.
            Look into having the raw data for input for spams.trainDL copied in.
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy

    result_array_size = X.shape[0] * kwargs["K"]
    result_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.array(0, dtype=X.dtype)))

    result_array = multiprocessing.Array(result_array_ctype,
                                         result_array_size)

    p = multiprocessing.Process(target = run_multiprocessing_array_spams_trainDL, args = (result_array, X,) + args, kwargs = kwargs)
    p.start()
    p.join()

    if p.exitcode != 0:
        raise SPAMSException("SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\".")

    result = numpy.frombuffer(result_array.get_obj(), dtype = result_array_ctype).reshape((-1, kwargs["K"]))
    result = result.copy()

    return(result)


#@nanshe.advanced_debugging.log_call(logger)
def call_spams_trainDL(*args, **kwargs):
    """
        Encapsulates call to spams.trainDL. Ensures copy of results occur just in case.
        Designed to be like the multiprocessing calls.

        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            For legacy.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    result = result.copy()

    return(result)