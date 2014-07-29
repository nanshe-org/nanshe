# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 14, 2014 5:42:07PM$"

import logging
import traceback
import functools
import os
import sys


# Nothing fancy. Just the basic logging unless otherwise specified, in which case this does nothing.
logging.basicConfig(level = logging.DEBUG, stream=sys.stderr)


def log_call(logger, to_log_call = True, to_print_args = False, to_print_exception = False):
    """
        Takes a given logger and uses it to log entering and leaving the decorated callable.
        Intended to be used as a decorator that takes a few arguments.

        Args:
            logger      (Logger):          Used for logging entry, exit and possibly arguments.
        
        Keyword Args:
            to_log_call (bool):            Whether to log call or not. This overrides all other arguments. It will be
                                           stored as a global variable on the function, which can be changed at runtime.

            to_print_args  (bool):         Whether to output the arguments and keyword arguments passed to the function.
                                           This should not automatically be true as some arguments may not be printable
                                           or may be expensive to print. Thus, it should be up to the developer to use
                                           their own discretion. Further, we don't want to break their existing code.

            to_print_exception  (bool):    Whether to print the traceback when an exception is raise.
        
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

            result = None
            # This allows keyword arguments to be turned on or off at runtime.
            if log_call_callable_wrapped.to_log_call:
                # Log that we have entered the callable in question.
                logger.debug("Entering callable: \"" + callable.__name__ + "\".")

                # Output arguments and keyword arguments if acceptable.
                # This allows keyword arguments to be turned on or off at runtime.
                #
                # Note: We have used log_call_callable_wrapped.to_print_args.
                #       However, we cannot define this until after as wrapping will lose this variable.
                if (log_call_callable_wrapped.to_print_args):
                    logger.debug("Arguments: \"" + str(args) + "\"" + os.linesep + "Keyword Arguments: \"" + str(kwargs) + "\".")

                # We don't return immediately. Why? We want to know if this succeeded or failed.
                # So, we want the log message below to print after the function runs.
                if log_call_callable_wrapped.to_print_exception:
                    try:
                        result = callable(*args, **kwargs)
                    except:
                        logger.error(traceback.format_exc())
                        raise
                else:
                    result = callable(*args, **kwargs)

                # Log that we have exited the callable in question.
                logger.debug("Exiting \"" + callable.__name__ + "\".")
            else:
                # We don't return immediately. Why? We want to know if this succeeded or failed.
                # So, we want the log message below to print after the function runs.
                result = callable(*args, **kwargs)


            # Return the result even None.
            return(result)

        # Store the underlying callable. Automatic in Python 3.
        log_call_callable_wrapped.__wrapped__ = callable

        # Copy over the value of to_log_call for later use.
        # Must be defined afterwards as functools.wraps will not copy it over to the wrapped instance.
        log_call_callable_wrapped.to_log_call = to_log_call

        # Copy over the value of to_print_args for later use.
        # Must be defined afterwards as functools.wraps will not copy it over to the wrapped instance.
        log_call_callable_wrapped.to_print_args = to_print_args

        # Copy over the value of to_print_exception for later use.
        # Must be defined afterwards as functools.wraps will not copy it over to the wrapped instance.
        log_call_callable_wrapped.to_print_exception = to_print_exception

        # The callable wrapped.
        return(log_call_callable_wrapped)

    # The arguments passed to the decorator for easy access.
    return(log_call_decorator)