# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$May 1, 2014 2:23:45 PM$"


import numpy



def estimate_noise(input_array, significance_threshhold = 3.0):
    """
        Estimates the noise in the given array.
        
        Using the array finds what the standard deviation is of some values in
        the array, which are within the standard deviation of the whole array times
        the significance threshhold
        
        Args:
            input_array(numpy.ndarray):          the array to estimate noise of.
            significance_threshhold(float):      the number of standard deviations (of the whole array), below which must be noise.
        
        Returns:
            cs(numpy.ndarray): a numpy array containing the row of Pascal's triangle.
        
        
        Examples:
            >>> estimate_noise(numpy.eye(2))
            0.5
            
            >>> estimate_noise(numpy.eye(2), significance_threshhold = 3)
            0.5
                   
            >>> estimate_noise(numpy.eye(3), significance_threshhold = 3)
            0.47140452079103173
            
            >>> import math; math.floor(1000*estimate_noise(numpy.random.random((2000,2000)), significance_threshhold = 1))/1000
            0.166
            
            >>> import math; math.floor(1000*estimate_noise(numpy.random.random((2000,2000)), significance_threshhold = 2))/1000
            0.288
            
            >>> import math; math.floor(1000*estimate_noise(numpy.random.random((2000,2000)), significance_threshhold = 3))/1000
            0.288
    """
    
    mean = input_array.mean()
    stddev = input_array.std()
    
    # Find cells that are inside an acceptable range (3 std devs from the mean by default)
    insignificant_mask = numpy.abs(input_array - mean) < significance_threshhold * stddev
    
    # Those cells have noise. Estimate the standard deviation on them. That will be our noise unit size.
    noise = input_array[insignificant_mask].std()
    
    return(noise)


def significant_mask(input_array, significance_threshhold = 3.0, noise_threshhold = 6.0):
    """
        Using estimate_noise, creates a mask that selects the non-noise and suppresses noise.
        
        Args:
            input_array(numpy.ndarray):          the array, which needs noise removed.
            significance_threshhold(float):      the number of standard deviations (of the whole array), below which must be noise.
            noise_threshhold(float):             the estimated noise times this value determines the max value to consider as noise (to zero).
            in_place(bool):                      whether to modify input_array directly or to return a copy instead.
        
        Returns:
            result(numpy.ndarray): a numpy array with noise zeroed.
        
        
        Examples:
            
            >>> significant_mask(numpy.eye(2), significance_threshhold = 3.0, noise_threshhold = 6.0)
            array([[False, False],
                   [False, False]], dtype=bool)
            
            >>> significant_mask(numpy.eye(2), significance_threshhold = 3.0, noise_threshhold = 1.0)
            array([[ True,  True],
                   [ True,  True]], dtype=bool)
                   
            >>> significant_mask(numpy.eye(3), significance_threshhold = 3.0, noise_threshhold = 2.0)
            array([[False, False, False],
                   [False, False, False],
                   [False, False, False]], dtype=bool)
            
            >>> significant_mask(numpy.eye(3), significance_threshhold = 3.0, noise_threshhold = 1.0)
            array([[ True, False, False],
                   [False,  True, False],
                   [False, False,  True]], dtype=bool)
    """
    
    mean = input_array.mean()
    
    # Those cells have noise. Estimate the standard deviation on them. That will be our noise unit size.
    noise = estimate_noise(input_array, significance_threshhold)
    
    # Get all the noisy points in a mask and toss them.
    significant_mask = numpy.abs(input_array - mean) >= noise_threshhold * noise
    
    return(significant_mask)


def noise_mask(input_array, significance_threshhold = 3.0, noise_threshhold = 6.0):
    """
        Using estimate_noise, creates a mask that selects the noise and suppresses non-noise.
        
        Args:
            input_array(numpy.ndarray):          the array to use for generating the noise mask.
            significance_threshhold(float):      the number of standard deviations (of the whole array), below which must be noise.
            noise_threshhold(float):             the estimated noise times this value determines the max value to consider as noise (to zero).
        
        Returns:
            result(numpy.ndarray): a numpy array with noise zeroed.
        
        
        Examples:
            >>> noise_mask(numpy.eye(2), significance_threshhold = 3.0, noise_threshhold = 6.0)
            array([[ True,  True],
                   [ True,  True]], dtype=bool)
            
            >>> noise_mask(numpy.eye(2), significance_threshhold = 3.0, noise_threshhold = 1.0)
            array([[False, False],
                   [False, False]], dtype=bool)
                   
            >>> noise_mask(numpy.eye(3), significance_threshhold = 3.0, noise_threshhold = 2.0)
            array([[ True,  True,  True],
                   [ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)
            
            >>> noise_mask(numpy.eye(3), significance_threshhold = 3.0, noise_threshhold = 1.0)
            array([[False,  True,  True],
                   [ True, False,  True],
                   [ True,  True, False]], dtype=bool)
    """
    
    # Get all the significant points in a mask.
    noisy_mask = significant_mask(input_array, significance_threshhold = significance_threshhold, noise_threshhold = noise_threshhold)
    
    # Invert the maske
    numpy.logical_not(noisy_mask, noisy_mask)
    
    return(noisy_mask)


def remove_noise(input_array, significance_threshhold = 3.0, noise_threshhold = 6.0, output_array = None):
    """
        Using estimate_noise, removes noise from the given array.
        
        Args:
            input_array(numpy.ndarray):          the array, which needs noise removed.
            significance_threshhold(float):      the number of standard deviations (of the whole array), below which must be noise.
            noise_threshhold(float):             the estimated noise times this value determines the max value to consider as noise (to zero).
            output_array(bool):                      whether to modify input_array directly or to return a copy instead.
        
        Returns:
            result(numpy.ndarray): a numpy array with noise zeroed.
        
        
        Examples:
            >>> remove_noise(numpy.eye(2), significance_threshhold = 3.0, noise_threshhold = 6.0)
            array([[ 0.,  0.],
                   [ 0.,  0.]])
            
            >>> remove_noise(numpy.random.random((2,2)), significance_threshhold = 3.0, noise_threshhold = 6.0)
            array([[ 0.,  0.],
                   [ 0.,  0.]])
            
            >>> a = numpy.random.random((2,2)); numpy.all(a != remove_noise(a, significance_threshhold = 3.0, noise_threshhold = 6.0))
            True
            
            >>> a = numpy.random.random((2,2)); numpy.all(a == remove_noise(a, significance_threshhold = 3.0, noise_threshhold = 6.0, output_array = a))
            True
    """
    
    # Get all the noisy points in a mask.
    sig_mask = significant_mask(input_array, significance_threshhold = significance_threshhold, noise_threshhold = noise_threshhold)
    
    if output_array is None:
        # No storage specified. So, use a zeroed array of the same shape and type as the input_array.
        output_array = numpy.zeros(input_array.shape).astype(input_array.dtype.type)
    else:
        # Storage 
        noisy_mask = numpy.logical_not(sig_mask)
        output_array[noisy_mask] = 0
    
    output_array[sig_mask] = input_array[sig_mask]
    
    return(output_array)