# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$Apr 14, 2014 5:42:07 PM$"


import logging
import functools


# Nothing fancy. Just the basic logging unless otherwise specified, in which case this does nothing.
logging.basicConfig(level=logging.DEBUG)


def log_call(logger, print_args = False):
    """
        Takes a given logger and uses it to log entering and leaving the decorated callable.
        
        Intended to be used as a decorator that takes a few arguments.
        
        Args:
            logger      (Logger):    Used for logging entry, exit and possibly arguments.
        
        Keyword Args:
            print_args  (bool):      Whether to output the arguments and keyword arguments passed to the function.
                                     This should not automatically be true as some arguments may not be printable or
                                     may be expensive to print. Thus, it should be up to the developer to use their
                                     own discretion. Further, we don't want to break their existing code.
        
        Returns:
            log_call_decorator (for wrapping)
    """
    
    def log_call_decorator(callable):
        """
            The actual decorator, which is what is returned to decorate the callable in question.

            Args:
                callable:    Anything that can be called function, method, functor, lambda, etc.
        
            Returns:
		log_call_callable_wrapped, which is wrapped around the function in question.
        """
        
        @functools.wraps(callable)
        def log_call_callable_wrapped(*args, **kwargs):
            """
                This is what will replace the orignal callable. It will behave the same except it will now log its
                entrance and exit. If set, it will log its parameters too.

                Args:
                    callable:    Anything that can be called function, method, functor, lambda, etc.

                Returns:
                    log_call_callable_wrapped, which is wrapped around the function in question.
            """
            
            # Log that we have entered the callable in question.
            logger.debug("Entering callable: \"" + callable.__name__ + "\".")
            
            # Output arguments and keyword arguments if acceptable. Note that this allows keyword arguments to be turned on or off at runtime.
            #
            # Note: We have used log_call_callable_wrapped.print_args. However, we cannot define this until after as wrapping will lose this variable.
            if (log_call_callable_wrapped.print_args):
               logger.debug("Called with the arguments: \"" + str(args) + "\" and with the keyword arguments: \"" + str(kwargs) + "\".")
            
            # We don't return immediately. Why? We want to know if this succeeded or failed.
            # So, we want the log message below to print after the function runs.
            result = callable(*args, **kwargs)
            
            # Log that we have exited the callable in question.
            logger.debug("Exiting " + callable.__name__ + ".")
            
            # Return the result even None.
            return(result)
        
        # Store the underlying callable. Automatic in Python 3.
        log_call_callable_wrapped.__wrapped__ = callable
        
        # Copy over the define value of print_args for later use.
        # Must be defined afterwards as functools.wraps will not copy it over to the wrapped instance.
        log_call_callable_wrapped.print_args = print_args
        
        # The callable wrapped.
        return(log_call_callable_wrapped)
    
    # The arguments passed to the decorator for easy access.
    return(log_call_decorator)