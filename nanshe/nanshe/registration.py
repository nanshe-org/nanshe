__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jan 28, 2015 11:25:47 EST$"



import warnings

import numpy
import bottleneck

import expanded_numpy

# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)



@debugging_tools.log_call(logger)
def dtfreg_fast_mean(frames2reg):
    """
        This algorithm registers the given image stack against its mean projection. This is done by computing
        translations needed to put each frame in alignment. Then the translation is performed and new translations are
        computed. This is repeated until no further improvement can be made.

        The code for translations can be found in dtfreg_fast_shifts.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           Image stack to register (time is the first dimension uses C-order tyx
                                                 or tzyx).

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

            >>> dtfreg_fast_mean(a)
            array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0]])
    """

    # Convert to a floating type if it isn't one.
    if not issubclass(frames2reg.dtype.type, numpy.floating):
        warnings.warn(
            "Provided new_data with type \"" + repr(frames2reg.dtype.type) +
            "\", which is not a floating type. Will be cast to type \"" + repr(numpy.float32) + "\"",
            RuntimeWarning
        )
        frames2reg = frames2reg.astype(numpy.float32)

    spaceShift = numpy.zeros((len(frames2reg), frames2reg.ndim-1), dtype=int)

    reg_frames = frames2reg.copy()
    reg_frames = reg_frames.view(numpy.ma.MaskedArray)
    reg_frames.mask = numpy.ma.getmaskarray(reg_frames)

    frames2reg_fft = numpy.fft.fftn(frames2reg, axes=range(1, frames2reg.ndim))
    template_fft = numpy.empty(frames2reg.shape[1:], dtype=complex)

    # Repeat shift calculation until there is no further adjustment.
    SSE = 1
    while SSE:
        template_fft[:] = numpy.conj(numpy.fft.fftn(reg_frames.mean(axis=0)))

        this_spaceShift = dtfreg_fast_shifts(frames2reg_fft, template_fft)

        # Adjust the registered frames using the translations found.
        # Mask rolled values.
        for i in xrange(len(reg_frames)):
            reg_frames[i] = expanded_numpy.roll(frames2reg[i], this_spaceShift[i], to_mask=True)

        SSE = ((this_spaceShift - spaceShift)**2).sum()

        spaceShift[:] = this_spaceShift

    return(spaceShift)


@debugging_tools.log_call(logger)
def dtfreg_fast_shifts(frames2reg_fft, template_fft):
    """
        Computes the convolution of the template with the frames by taking advantage of their FFTs for faster
        computation that an ordinary convolution ( O(N*lg(N)) vs O(N^2) )
        < https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution_vs_Direct.html >.
        Once computed the maximum of the convolution is found to determine the best overlap of each frame with the
        template, which provides the needed offset. Some corrections are performed to make reasonable offsets.

        Notes:
            Adapted from code provided by Wenzhi Sun with speed improvements provided by Uri Dubin.

        Args:
            frames2reg(numpy.ndarray):           image stack to register (time is the first dimension uses C-order tyx
                                                 or tzyx).

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

            >>> tf = numpy.conj(numpy.fft.fftn(a.mean(axis=0))); tf
            array([[ 4.0-0.j        ,  0.0-0.j        ,  0.0-0.j        ,  0.0-0.j        ],
                   [ 2.8-0.69282032j,  0.0-0.j        ,  0.0-0.j        ,  0.0-0.j        ],
                   [ 2.8+0.69282032j,  0.0-0.j        ,  0.0-0.j        ,  0.0-0.j        ]])

            >>> dtfreg_fast_shifts(af, tf)
            array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0],
                   [0, 0]])
    """

    # Compute the product of the two FFTs (i.e. the convolution of the regular versions).
    frames2reg_template_conv_fft = frames2reg_fft * template_fft[None]

    # Find the FFT inverse (over all spatial dimensions) to return to the convolution.
    frames2reg_template_conv = numpy.fft.ifftn(frames2reg_template_conv_fft, axes=range(1, frames2reg_fft.ndim))

    # Find where the convolution is maximal. Will have the most things in common between the template and frames.
    frames2reg_template_conv_max, frames2reg_template_conv_max_indices = expanded_numpy.max_abs(
        frames2reg_template_conv, axis=range(1, frames2reg_fft.ndim), return_indices=True
    )

    # First index is just the frame, which will be in sequential order. We don't need this so we drop it.
    frames2reg_template_conv_max_indices = frames2reg_template_conv_max_indices[1:]

    # Convert indices into an array for easy manipulation.
    frames2reg_template_conv_max_indices = numpy.array(frames2reg_template_conv_max_indices).T.copy()

    # Get the shape and half shape of the spatial components (YX or ZYX) for easy manipulation.
    frames2reg_fft_spatial_shape = numpy.array(frames2reg_fft.shape[1:])
    frames2reg_fft_spatial_half_shape = numpy.trunc(frames2reg_fft_spatial_shape/2.0)

    # Remove global shifts.
    frames2reg_template_conv_max_indices_offset = numpy.trunc(
        frames2reg_template_conv_max_indices.mean(axis=0)
    ).astype(int)
    frames2reg_template_conv_max_indices -= frames2reg_template_conv_max_indices_offset[None]

    # Find the shortest roll possible (i.e. if it is going over halfway switch direction so it will go less than half).
    # Note all indices by definition were positive semi-definite and upper bounded by the shape. This change will make
    # them bound by the half shape, but with either sign.
    frames2reg_template_conv_max_indices_mask = (
        frames2reg_template_conv_max_indices > frames2reg_fft_spatial_half_shape[None]
    )
    if frames2reg_template_conv_max_indices_mask.any():
        frames2reg_template_conv_max_indices -= frames2reg_template_conv_max_indices_mask*frames2reg_fft_spatial_shape

    # Shift will have to be in the opposite direction to bring everything to the center.
    numpy.negative(frames2reg_template_conv_max_indices, out=frames2reg_template_conv_max_indices)

    return(frames2reg_template_conv_max_indices)
