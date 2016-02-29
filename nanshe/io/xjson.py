"""
The module ``xjson`` serializes JSON with some additional features.

===============================================================================
Overview
===============================================================================
The module ``xjson`` provides a mechanism of serializing JSON in a way that
allows for a few additional constraints.

- Commenting -- does not break the JSON specification (e.g. adding the prefix\
  ``__comment__`` to any string).
- Order dependent deserialization -- dictionary order is preserved.
- ASCII strings -- all strings are converted to ASCII

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 30, 2014 16:54:30 EDT$"


# Need in order to have logging information no matter what.
from nanshe.util import prof


try:
    unicode
except NameError:
    unicode = str


# Get the logger
trace_logger = prof.getTraceLogger(__name__)
logger = prof.logging.getLogger(__name__)


@prof.log_call(trace_logger)
def read_parameters(config_filename, maintain_order=False):
    """
        Reads the contents of a json config file and returns the parameters.

        Args:
            config_filename (str):                          name of the file
                                                            to read.

            maintain_order (bool):                          whether to preserve
                                                            the order of keys
                                                            in the json file

        Returns:
            parameters (dict or collections.OrderedDict):   parameters read
                                                            from the file.
    """


    # only relevant if reading parameter file.
    import json

    # Get the type of dictionary to use.
    json_dict = None
    if maintain_order:
        # only relevant if reading parameter file and maintaining order.
        import collections
        json_dict = collections.OrderedDict
    else:
        json_dict = dict


    @prof.log_call(trace_logger)
    def ascii_encode_str(value, json_dict=json_dict):
        """
            Encodes the str.

            Args:
                value(str):     string to encode.

            Returns:
                str:            Properly encoded str.
        """

        new_value = None
        if unicode == str and not value.startswith(u"__comment__"):
            new_value = value
        elif not value.startswith("__comment__"):
            new_value = value.encode("utf-8")

        return(new_value)


    @prof.log_call(trace_logger)
    def ascii_encode_list(data, json_dict=json_dict):
        """
            Encodes the list (and its contents).

            Args:
                data(list):      list to encode.

            Returns:
                list:            Properly encoded list.
        """
        transformed_list = []

        for each_value in data:
            new_each_value = each_value
            if isinstance(new_each_value, json_dict):
                new_each_value = ascii_encode_dict(new_each_value)
            elif isinstance(new_each_value, list):
                new_each_value = ascii_encode_list(new_each_value)
            elif isinstance(new_each_value, (bytes, unicode)):
                new_each_value = ascii_encode_str(new_each_value)

            if new_each_value is not None:
                transformed_list.append(new_each_value)

        return(transformed_list)


    @prof.log_call(trace_logger)
    def ascii_encode_dict(data, json_dict=json_dict):
        """
            Encodes the dict (and its contents).
            Also, make sure the dict is of the right type.

            Args:
                data(dict):      dict to encode.

            Returns:
                dict:            Properly encoded dict.
        """
        new_dict = json_dict(data)

        transformed_dict = []

        for each_key, each_value in new_dict.items():
            new_each_key = ascii_encode_str(each_key)

            new_each_value = each_value
            if new_each_key is not None:
                if isinstance(new_each_value, json_dict):
                    new_each_value = ascii_encode_dict(new_each_value)
                elif isinstance(new_each_value, list):
                    new_each_value = ascii_encode_list(new_each_value)
                elif isinstance(new_each_value, unicode) or \
                        isinstance(new_each_value, str):
                    new_each_value = ascii_encode_str(new_each_value)

                if new_each_value is not None:
                    transformed_dict.append((new_each_key, new_each_value))

        transformed_dict = json_dict(transformed_dict)

        return(transformed_dict)


    # gets parameters out of the file and dumps them in the dictionary. just
    # that simple.
    parameters = None
    with open(config_filename, 'r') as fp:
        logger.debug(
            "Opened configure file named \"" + config_filename + "\"."
        )

        # will just give a dictionary. just that simple
        parameters = json.load(fp, object_pairs_hook=ascii_encode_dict)

        logger.debug(
            "Loaded parameters from file, which are \"" +
            str(parameters) + "\"."
        )

    return(parameters)
