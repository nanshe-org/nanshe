"""
The ``registration`` module helps bring image stacks into alignment.

===============================================================================
Overview
===============================================================================
The registration algorithm developed by Wenzhi Sun. The strategy uses an
area-based method for registration. Namely, it takes a template image (the mean
projection) and finds the translations required for each frame to overlap
optimally with the template image. Then a number of adjustments are made to the
shifts to ensure that unnecessary translations are not preformed.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jan 28, 2015 11:25:47 EST$"


import itertools
import os
import tempfile
import warnings

import h5py
import numpy


try:
    import pyfftw.interfaces.numpy_fft as fft
except Exception as e:
    warnings.warn(str(e) + ". Falling back to NumPy FFTPACK.", ImportWarning)
    import numpy.fft as fft

from nanshe.util import iters, xnumpy
from nanshe.io import hdf5

# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the loggers
trace_logger = prof.getTraceLogger(__name__)
logger = prof.logging.getLogger(__name__)



@prof.log_call(trace_logger)
def register_mean_offsets(frames2reg, max_iters=-1, block_frame_length=-1, include_shift=False, to_truncate=False, float_type=numpy.dtype(float).type):
    """
        This algorithm registers the given image stack against its mean projection. This is done by computing
        translations needed to put each frame in alignment. Then the translation is performed and new translations are
        computed. This is repeated until no further improvement can be made.

        The code for translations can be found in find_mean_offsets.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           Image stack to register (time is the first dimension uses C-order tyx
                                                 or tzyx).
            max_iters(int):                      Number of iterations to allow before forcing termination if stable
                                                 point is not found yet. Set to -1 if no limit. (Default -1)
            block_frame_length(int):             Number of frames to work with at a time.
                                                 By default all. (Default -1)
            include_shift(bool):                 Whether to return the shifts used, as well. (Default False)
            to_truncate(bool):                   Whether to truncate the frames to remove all masked portions. (Default False)
            float_type(type):                    Type of float to use for calculation. (Default numpy.float64).

        Returns:
            (numpy.ndarray):                     an array containing the translations to apply to each frame.

        Examples:
            >>> a = numpy.zeros((5, 3, 4)); a[:,0] = 1; a[2,0] = 0; a[2,2] = 1; a
            array([[[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]]])

            >>> register_mean_offsets(a, include_shift=True)
            (masked_array(data =
             [[[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[-- -- -- --]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]
            <BLANKLINE>
             [[1.0 1.0 1.0 1.0]
              [0.0 0.0 0.0 0.0]
              [0.0 0.0 0.0 0.0]]],
                         mask =
             [[[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[ True  True  True  True]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]
            <BLANKLINE>
             [[False False False False]
              [False False False False]
              [False False False False]]],
                   fill_value = 0.0)
            , array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0]]))
    """

    float_type = numpy.dtype(float_type).type

    # Must be of type float and must be at least 32-bit (smallest complex type
    # uses two 32-bit floats).
    assert issubclass(float_type, numpy.floating)
    assert numpy.dtype(float_type).itemsize >= 4

    # Sadly, there is no easier way to map the two types; so, this is it.
    float_complex_mapping = {
        numpy.float32 : numpy.complex64,
        numpy.float64 : numpy.complex128,
        numpy.float128 : numpy.complex256
    }
    complex_type = float_complex_mapping[float_type]

    if block_frame_length == -1:
        block_frame_length = len(frames2reg)

    tempdir_name = ""
    temporaries_filename = ""
    if isinstance(frames2reg, h5py.Dataset):
        tempdir_name, temporaries_filename = os.path.split(
            os.path.abspath(frames2reg.file.filename)
        )

        temporaries_filename = os.path.splitext(temporaries_filename)[0]
        temporaries_filename += "_".join(
            [
                frames2reg.name.replace("/", "_"),
                "temporaries.h5"
            ]
        )
        temporaries_filename = os.path.join(
            tempdir_name,
            temporaries_filename
        )
    elif (block_frame_length != len(frames2reg)):
        tempdir_name = tempfile.mkdtemp()
        temporaries_filename = os.path.join(tempdir_name, "temporaries.h5")

    frames2reg_fft = None
    space_shift = None
    this_space_shift = None
    if tempdir_name:
        temporaries_file = h5py.File(temporaries_filename, "w")

        frames2reg_fft = temporaries_file.create_dataset(
            "frames2reg_fft", shape=frames2reg.shape, dtype=complex_type
        )
        space_shift = temporaries_file.create_dataset(
            "space_shift",
            shape=(len(frames2reg), len(frames2reg.shape)-1),
            dtype=int
        )
        this_space_shift = temporaries_file.create_dataset(
            "this_space_shift",
            shape=space_shift.shape,
            dtype=space_shift.dtype
        )
    else:
        frames2reg_fft = numpy.empty(frames2reg.shape, dtype=complex_type)
        space_shift = numpy.zeros(
            (len(frames2reg), len(frames2reg.shape)-1), dtype=int
        )
        this_space_shift = numpy.empty_like(space_shift)

    for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
        frames2reg_fft[i:j] = fft.fftn(frames2reg[i:j], axes=range(1, len(frames2reg.shape)))
    template_fft = numpy.empty(frames2reg.shape[1:], dtype=complex_type)

    negative_wave_vector = numpy.asarray(template_fft.shape, dtype=float_type)
    numpy.reciprocal(negative_wave_vector, out=negative_wave_vector)
    negative_wave_vector *= 2*numpy.pi
    numpy.negative(negative_wave_vector, out=negative_wave_vector)

    template_fft_indices = xnumpy.cartesian_product([numpy.arange(_) for _ in template_fft.shape])

    unit_space_shift_fft = template_fft_indices * negative_wave_vector
    unit_space_shift_fft = unit_space_shift_fft.T.copy()
    unit_space_shift_fft = unit_space_shift_fft.reshape((template_fft.ndim,) + template_fft.shape)

    negative_wave_vector = None
    template_fft_indices = None

    # Repeat shift calculation until there is no further adjustment.
    num_iters = 0
    squared_magnitude_delta_space_shift = 1.0
    while (squared_magnitude_delta_space_shift != 0.0):
        squared_magnitude_delta_space_shift = 0.0

        template_fft[:] = 0
        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            frames2reg_shifted_fft_ij = numpy.exp(1j * numpy.tensordot(space_shift[i:j], unit_space_shift_fft, axes=[-1, 0]))
            frames2reg_shifted_fft_ij *= frames2reg_fft[i:j]
            template_fft += numpy.sum(frames2reg_shifted_fft_ij, axis=0)
        template_fft /= len(frames2reg)

        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            this_space_shift[i:j] = find_offsets(frames2reg_fft[i:j], template_fft)

        # Remove global shifts.
        this_space_shift_mean = numpy.zeros(this_space_shift.shape[1:], dtype=this_space_shift.dtype)
        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            this_space_shift_mean = this_space_shift[i:j].sum(axis=0)
        this_space_shift_mean = numpy.round(
            this_space_shift_mean.astype(float_type) / len(this_space_shift)
        ).astype(int)
        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            this_space_shift[i:j] = xnumpy.find_relative_offsets(
                this_space_shift[i:j],
                this_space_shift_mean
            )

        # Find the shortest roll possible (i.e. if it is going over halfway switch direction so it will go less than half).
        # Note all indices by definition were positive semi-definite and upper bounded by the shape. This change will make
        # them bound by the half shape, but with either sign.
        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            this_space_shift[i:j] = xnumpy.find_shortest_wraparound(
                this_space_shift[i:j],
                frames2reg_fft.shape[1:]
            )

        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            delta_space_shift_ij = this_space_shift[i:j] - space_shift[i:j]
            squared_magnitude_delta_space_shift += numpy.dot(
                delta_space_shift_ij, delta_space_shift_ij.T
            ).sum()

        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            space_shift[i:j] = this_space_shift[i:j]

        num_iters += 1
        logger.info(
            "Completed iteration, %i, " %
                num_iters
            + "where the L_2 norm squared of the relative shift was, %f." %
                squared_magnitude_delta_space_shift
        )
        if max_iters != -1:
            if num_iters >= max_iters:
                logger.info("Hit maximum number of iterations.")
                break

    reg_frames_shape = frames2reg.shape
    if to_truncate:
        space_shift_max = numpy.zeros(
            space_shift.shape[1:], dtype=space_shift.dtype
        )
        space_shift_min = numpy.zeros(
            space_shift.shape[1:], dtype=space_shift.dtype
        )
        for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
            numpy.maximum(
                space_shift_max,
                space_shift[i:j].max(axis=0),
                out=space_shift_max
            )
            numpy.minimum(
                space_shift_min,
                space_shift[i:j].min(axis=0),
                out=space_shift_min
            )
        reg_frames_shape = numpy.asarray(reg_frames_shape)
        reg_frames_shape[1:] -= space_shift_max
        reg_frames_shape[1:] += space_shift_min
        reg_frames_shape = tuple(reg_frames_shape)

        space_shift_max = tuple(space_shift_max)
        space_shift_min = space_shift_min.astype(object)
        space_shift_min[space_shift_min == 0] = None
        space_shift_min = tuple(space_shift_min)
        reg_frames_slice = tuple(slice(_1, _2) for _1, _2 in itertools.izip(space_shift_max, space_shift_min))

    # Adjust the registered frames using the translations found.
    # Mask rolled values.
    reg_frames = None
    if tempdir_name:
        if to_truncate:
            reg_frames = temporaries_file.create_dataset(
                "reg_frames",
                shape=reg_frames_shape,
                dtype=frames2reg.dtype,
                chunks=True
            )
        else:
            reg_frames = temporaries_file.create_group("reg_frames")
            reg_frames = hdf5.serializers.HDF5MaskedDataset(
                reg_frames, shape=frames2reg.shape, dtype=frames2reg.dtype
            )
    else:
        if to_truncate:
            reg_frames = numpy.empty(reg_frames_shape, dtype=frames2reg.dtype)
        else:
            reg_frames = numpy.ma.empty_like(frames2reg)
            reg_frames.mask = numpy.ma.getmaskarray(reg_frames)
            reg_frames.set_fill_value(reg_frames.dtype.type(0))

    for i, j in iters.lagged_generators_zipped(itertools.chain(xrange(0, len(frames2reg), block_frame_length), [len(frames2reg)])):
        for k in xrange(i, j):
            if to_truncate:
                reg_frames[k] = xnumpy.roll(frames2reg[k], space_shift[k])[reg_frames_slice]
            else:
                reg_frames[k] = xnumpy.roll(frames2reg[k], space_shift[k], to_mask=True)

    result = None
    results_filename = ""
    if tempdir_name:
        result = results_filename
        results_filename = os.path.join(tempdir_name, "results.h5")
        results_file = h5py.File(results_filename, "w")
        if to_truncate:
            temporaries_file.copy(reg_frames.name, results_file)
        else:
            temporaries_file.copy(reg_frames.group, results_file)
        if include_shift:
            temporaries_file.copy(space_shift, results_file)
        frames2reg_fft = None
        reg_frames = None
        space_shift = None
        this_space_shift = None
        temporaries_file.close()
        os.remove(temporaries_filename)
        temporaries_filename = ""
        result = results_filename
    else:
        result = reg_frames
        if include_shift:
            result = (reg_frames, space_shift)

    if tempdir_name:
        results_file.close()
        results_file = None

    return(result)


@prof.log_call(trace_logger)
def find_offsets(frames2reg_fft, template_fft):
    """
        Computes the convolution of the template with the frames by taking advantage of their FFTs for faster
        computation that an ordinary convolution ( O(N*lg(N)) vs O(N^2) )
        < http://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution_vs_Direct.html >.
        Once computed the maximum of the convolution is found to determine the best overlap of each frame with the
        template, which provides the needed offset. Some corrections are performed to make reasonable offsets.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           image stack to register (time is the first dimension uses C-order tyx
                                                 or tzyx).
            template_fft(numpy.ndarray):         what to register the image stack against (single frame using C-order
                                                 yx or zyx).

        Returns:
            (numpy.ndarray):                     an array containing the translations to apply to each frame.

        Examples:
            >>> a = numpy.zeros((5, 3, 4)); a[:,0] = 1; a[2,0] = 0; a[2,2] = 1; a
            array([[[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]],
            <BLANKLINE>
                   [[ 1.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.]]])

            >>> af = numpy.fft.fftn(a, axes=range(1, a.ndim)); af
            array([[[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [-2.+3.46410162j,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [-2.-3.46410162j,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]],
            <BLANKLINE>
                   [[ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ],
                    [ 4.+0.j        ,  0.+0.j        ,  0.+0.j        ,  0.+0.j        ]]])

            >>> tf = numpy.fft.fftn(a.mean(axis=0)); tf
            array([[ 4.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ],
                   [ 2.8+0.69282032j,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ],
                   [ 2.8-0.69282032j,  0.0+0.j        ,  0.0+0.j        ,  0.0+0.j        ]])

            >>> find_offsets(af, tf)
            array([[ 0,  0],
                   [ 0,  0],
                   [-2,  0],
                   [ 0,  0],
                   [ 0,  0]])
    """

    # If there is only one frame, add a singleton axis to indicate this.
    frames2reg_fft_added_singleton = (frames2reg_fft.ndim == template_fft.ndim)
    if frames2reg_fft_added_singleton:
        frames2reg_fft = frames2reg_fft[None]

    # Compute the product of the two FFTs (i.e. the convolution of the regular versions).
    frames2reg_template_conv_fft = frames2reg_fft * template_fft.conj()[None]

    # Find the FFT inverse (over all spatial dimensions) to return to the convolution.
    frames2reg_template_conv = fft.ifftn(frames2reg_template_conv_fft, axes=range(1, frames2reg_fft.ndim))

    # Find where the convolution is maximal. Will have the most things in common between the template and frames.
    frames2reg_template_conv_max, frames2reg_template_conv_max_indices = xnumpy.max_abs(
        frames2reg_template_conv, axis=range(1, frames2reg_fft.ndim), return_indices=True
    )

    # First index is just the frame, which will be in sequential order. We don't need this so we drop it.
    frames2reg_template_conv_max_indices = frames2reg_template_conv_max_indices[1:]

    # Convert indices into an array for easy manipulation.
    frames2reg_template_conv_max_indices = numpy.array(frames2reg_template_conv_max_indices).T.copy()

    # Shift will have to be in the opposite direction to bring everything to the center.
    numpy.negative(frames2reg_template_conv_max_indices, out=frames2reg_template_conv_max_indices)

    return(frames2reg_template_conv_max_indices)
