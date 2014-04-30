# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$Apr 30, 2014 5:14:50 PM$"



# Generally useful and fast to import so done immediately.
import numpy

# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def zeroed_mean_images(new_numpy_array):
    """
        Takes and finds the mean for each image. Where each image is new_numpy_array[i] with some index i.
        
        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the first index
        
        Returns:
            result(numpy.ndarray):              The same array with each images mean removed. Where means[i] = mean(new_numpy_array[i])
        
        
        Examples:
            >>> import numpy; zeroed_mean_images(numpy.array([[0,0],[0,0]]))
            array([[ 0.,  0.],
                   [ 0.,  0.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[6,0],[0,0]]))
            array([[ 3., -3.],
                   [ 0.,  0.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[0,0],[0,4]]))
            array([[ 0.,  0.],
                   [-2.,  2.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[6,0],[0,4]]))
            array([[ 3., -3.],
                   [-2.,  2.]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[1,2],[3,4]]))
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
                   
            >>> import numpy; zeroed_mean_images(numpy.array([[1,2],[3,4]]))
            array([[-0.5,  0.5],
                   [-0.5,  0.5]])
    """
    
    # start with means having the same contents as the given images
    means = new_numpy_array
    
    # take the mean while we haven't gotten one mean for each image.
    while means.ndim > 1:
        means = means.mean(axis = 1)
    
    # reshape means until it has the right number of dimensions to broadcast.
    while means.ndim < new_numpy_array.ndim:
        means = means.reshape(means.shape + (1,))
    
    # broadcast and subtract the means so that the mean of all values in result[i] is zero
    result = new_numpy_array - means
    
    return(result)


@advanced_debugging.log_call(logger)
def renormalized_images(new_numpy_array, ord = 2):
    """
        Takes and finds the mean for each image. Where each image is new_numpy_array[i] with some index i.
        
        Args:
            new_numpy_array(numpy.ndarray):     array images with time as the first index
        
        Returns:
            result(numpy.ndarray):              The same array with each images mean removed. Where means[i] = mean(new_numpy_array[i])
        
        
        Examples:
            >>> import numpy; renormalized_images(numpy.array([[0,1],[1,0]]))
            array([[0, 1],
                   [1, 0]])
                   
            >>> import numpy; renormalized_images(numpy.array([[0.,2.],[1.,0.]]))
            array([[ 0.,  1.],
                   [ 1.,  0.]])
                   
            >>> import numpy; renormalized_images(numpy.array([[2.,2.],[1.,0.]]))
            array([[ 0.70710678,  0.70710678],
                   [ 1.        ,  0.        ]])
                   
            >>> import numpy; renormalized_images(numpy.array([[1.,2.],[3.,4.]]))
            array([[ 0.4472136 ,  0.89442719],
                   [ 0.6       ,  0.8       ]])
                   
            >>> import numpy; renormalized_images(numpy.array([[1.,2.],[3.,4.]]), ord = 1)
            array([[ 0.33333333,  0.66666667],
                   [ 0.42857143,  0.57142857]])
    """
    
    # must copy as we will change contents
    result = new_numpy_array.copy()
    
    # take each image at each time turn the image into a vector and find the norm.
    # divide each image by this norm. (required for spams.trainDL)
    for i in xrange(result.shape[0]):
        result[i] /= numpy.linalg.norm(result[i].ravel(), ord = ord)
    
    return(result)