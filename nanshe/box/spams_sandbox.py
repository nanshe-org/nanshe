"""
The ``spams_sandbox`` module provides mechanisms for sandboxing SPAMS.

===============================================================================
Overview
===============================================================================
SPAMS sometimes seems to step on the interpreter. As a result, we provide a
number of strategies to address this, by launching it in a separate process so
that it hopefully does not mess up the main interpreter. We also try to keep
the module space clean. This seems to help.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 20, 2014 12:07:48 EDT$"


import npctypes
import npctypes.shared


class SPAMSException(Exception):
    pass


def run_multiprocessing_queue_spams_trainDL(out_queue, *args, **kwargs):
    """
        Designed to run spams.trainDL in a separate process.

        It is necessary to run SPAMS in a separate process as segmentation
        faults have been discovered in later parts of the Python code dependent
        on whether SPAMS has run or not. It is suspected that spams may
        interfere with the interpreter. Thus, it should be sandboxed (run in a
        different Python interpreter) so that it doesn't damage what happens in
        this one.

        This particular version uses a multiprocessing.Queue to return the
        resulting dictionary.


        Args:
            out_queue(multiprocessing.Queue):       what will take the returned
                                                    dictionary from
                                                    spams.trainDL.

            *args(list):                            a list of position
                                                    arguments to pass to
                                                    spams.trainDL.

            **kwargs(dict):                         a dictionary of keyword
                                                    arguments to pass to
                                                    spams.trainDL.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    out_queue.put(result)


def call_multiprocessing_queue_spams_trainDL(*args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the
        result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation
        faults have been discovered in later parts of the Python code dependent
        on whether SPAMS has run or not. It is suspected that spams may
        interfere with the interpreter. Thus, it should be sandboxed (run in a
        different Python interpreter) so that it doesn't damage what happens in
        this one.

        This particular version uses a multiprocessing.Queue to return the
        resulting dictionary.


        Args:
            *args(list):                            a list of position
                                                    arguments to pass to
                                                    spams.trainDL.

            **kwargs(dict):                         a dictionary of keyword
                                                    arguments to pass to
                                                    spams.trainDL.

        Returns:
            result(numpy.matrix): the dictionary found
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    out_queue = multiprocessing.Queue()

    queue_args = (out_queue,) + args
    p = multiprocessing.Process(
        target=run_multiprocessing_queue_spams_trainDL,
        args=queue_args,
        kwargs=kwargs
    )
    p.start()
    result = out_queue.get()
    result = result.copy()
    p.join()

    if p.exitcode != 0: raise SPAMSException(
        "SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\"."
    )

    return(result)


def run_multiprocessing_array_spams_trainDL(result_array_type,
                                            result_array,
                                            X_array_type,
                                            X_array,
                                            D_is_arg=False,
                                            D_array_type=None,
                                            D_array=None,
                                            *args,
                                            **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the
        result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation
        faults have been discovered in later parts of the Python code dependent
        on whether SPAMS has run or not. It is suspected that spams may
        interfere with the interpreter. Thus, it should be sandboxed (run in a
        different Python interpreter) so that it doesn't damage what happens in
        this one.

        This particular version uses a multiprocessing.Array to share memory to
        return the resulting dictionary.


        Args:
            result_array_type(numpy.ctypeslib.ndpointer):   Unused will drop.
                                                            A pointer type with
                                                            properties needed
                                                            by result_array.

            result_array(multiprocessing.RawArray):         shared memory array
                                                            to store results
                                                            in.

            X_array_type(numpy.ctypeslib.ndpointer):        Unused will drop.
                                                            a pointer type with
                                                            properties needed
                                                            by X_array.

            X_array(numpy.ndarray):                         currently uses
                                                            numpy ndarray as
                                                            input.

            D_is_arg(bool):                                 Whether D either is
                                                            an arg and/or
                                                            should be an arg.

            D_array_type(numpy.ctypeslib.ndpointer):        Unused will drop.
                                                            a pointer type with
                                                            properties needed
                                                            by D_array.

            D_array(numpy.ndarray):                         currently uses
                                                            numpy ndarray as
                                                            the initial
                                                            dictionary.

            *args(list):                                    a list of position
                                                            arguments to pass
                                                            to spams.trainDL.

            **kwargs(dict):                                 a dictionary of
                                                            keyword arguments
                                                            to pass to
                                                            spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.
    """

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    # Also, it is not needed outside of calling this function.
    import spams

    with npctypes.shared.as_ndarray(X_array) as X:
        with npctypes.shared.as_ndarray(result_array) as result:
            if D_array is not None:
                with npctypes.shared.as_ndarray(D_array) as D:
                    if D_is_arg:
                        args[3] = D
                    else:
                        kwargs["D"] = D

                    result[:] = spams.trainDL(X, *args, **kwargs)
            else:
                result[:] = spams.trainDL(X, *args, **kwargs)


def call_multiprocessing_array_spams_trainDL(X, *args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle
        result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation
        faults have been discovered in later parts of the Python code dependent
        on whether SPAMS has run or not. It is suspected that spams may
        interfere with the interpreter. Thus, it should be sandboxed (run in a
        different Python interpreter) so that it doesn't damage what happens in
        this one.

        This particular version uses a multiprocessing.Array to share memory to
        return the resulting dictionary.


        Args:
            X(numpy.matrix):                        a Fortran order NumPy
                                                    Matrix with the same name
                                                    as used by spams.trainDL
                                                    (so if someone tries to use
                                                    it as a keyword
                                                    argument...).

            *args(list):                            a list of position
                                                    arguments to pass to
                                                    spams.trainDL.

            **kwargs(dict):                         a dictionary of keyword
                                                    arguments to pass to
                                                    spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy

    D_is_arg = False
    D = None
    if (len(args) >= 4):
        D_is_arg = True
        D = args[3]
        args[3] = None
    else:
        D = kwargs.pop("D", None)

    # Create a shared array to contain X
    X_array = npctypes.shared.ndarray(X.shape, X.dtype, "F")

    # Copy over the contents of X.
    with npctypes.shared.as_ndarray(X_array) as X_array_numpy:
        X_array_numpy[...] = X
    del X_array_numpy

    len_D = kwargs.get("K", None)
    if D is not None:
        # Create a shared array to contain D
        D_array = npctypes.shared.ndarray(D.shape, D.dtype, "F")

        # Copy over the contents of D.
        with npctypes.shared.as_ndarray(D_array) as D_array_numpy:
            D_array_numpy[...] = D
        del D_array_numpy

        len_D = D.shape[-1]

    # Create a shared array to contain the result
    result_array = npctypes.shared.ndarray((X.shape[0], len_D), X.dtype, "F")

    new_args = (
        type(result_array),
        result_array,
        type(X_array),
        X_array,
    )
    if D is not None:
        new_args = new_args + (
            D_is_arg, type(D_array), D_array,
        )
    p = multiprocessing.Process(
        target=run_multiprocessing_array_spams_trainDL,
        args=new_args,
        kwargs=kwargs
    )
    p.start()
    p.join()

    if p.exitcode != 0: raise SPAMSException(
        "SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\"."
    )

    # Reconstruct the result from the output array
    result = None
    with npctypes.shared.as_ndarray(result_array) as result:
        result = result.copy()

    return(result)


def call_spams_trainDL(*args, **kwargs):
    """
        Encapsulates call to spams.trainDL. Ensures copy of results occur just
        in case. Designed to be like the multiprocessing calls.

        Args:
            *args(list):                            a list of position
                                                    arguments to pass to
                                                    spams.trainDL.

            **kwargs(dict):                         a dictionary of keyword
                                                    arguments to pass to
                                                    spams.trainDL.

        Note:
            For legacy.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    result = result.copy()

    return(result)
