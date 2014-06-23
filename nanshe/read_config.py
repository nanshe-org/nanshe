# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 30, 2014 4:54:30PM$"


# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def read_parameters(config_filename, maintain_order = False):
    """
        Reads the contents of a json config file and returns the parameters.
        
        Args:
            config_filename (str):                          name of the file to read.
            maintain_order (bool):                          whether to preserve the order of keys in the json file
        
        Returns:
            parameters (dict or collections.OrderedDict):   parameters read from the file.
    """


    # only relevant if reading parameter file.
    import json

    # Get the type of dictionary to use.
    json_dict = None
    if maintain_order:
        # only relevant if reading parameter file and maintaing order.
        import collections
        json_dict = collections.OrderedDict
    else:
        json_dict = dict


    @advanced_debugging.log_call(logger)
    def ascii_encode_str(data, json_dict = json_dict):
        """
            Encodes the str.

            Args:
                data(str):      string to encode.

            Returns:
                str:            Properly encoded str.
        """
        return(data.encode("utf-8"))


    @advanced_debugging.log_call(logger)
    def ascii_encode_list(data, json_dict = json_dict):
        """
            Encodes the list (and its contents).

            Args:
                data(list):      list to encode.

            Returns:
                list:            Properly encoded list.
        """
        transformed_list = []

        for each_value in data:
            if isinstance(each_value, json_dict):
                each_value = ascii_encode_dict(each_value)

                # Drop comments from dictionaries
                new_each_value = json_dict()
                for each_key_in_each_value, each_value_in_each_value in each_value.items():
                    if not each_key_in_each_value.startswith("__comment__"):
                        new_each_value[each_key_in_each_value] = each_value_in_each_value
                each_value = new_each_value
            elif isinstance(each_value, list):
                each_value = ascii_encode_list(each_value)

                # Drop comments from lists
                new_each_value = list()
                for each_value_in_each_value in each_value.items():
                    if not each_value_in_each_value.startswith("__comment__"):
                        new_each_value.append(each_value_in_each_value)
                each_value = new_each_value
            elif isinstance(each_value, unicode) or isinstance(each_value, str):
                each_value = ascii_encode_str(each_value)

            transformed_list.append( each_value )

        return(transformed_list)


    @advanced_debugging.log_call(logger)
    def ascii_encode_dict(data, json_dict = json_dict):
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
            each_key = each_key.encode("utf-8")

            if isinstance(each_value, json_dict):
                each_value = ascii_encode_dict(each_value)
            elif isinstance(each_value, list):
                each_value = ascii_encode_list(each_value)
            elif isinstance(each_value, unicode) or isinstance(each_value, str):
                each_value = ascii_encode_str(each_value)

            transformed_dict.append( (each_key, each_value) )

        transformed_dict = json_dict(transformed_dict)

        return(transformed_dict)


    # gets parameters out of the file and dumps them in the dictionary. just that simple.
    parameters = None
    with open(config_filename, 'r') as fp:
        logger.debug("Opened configure file named \"" + config_filename + "\".")

        # will just give a dictionary. just that simple
        parameters = json.load(fp, object_pairs_hook = ascii_encode_dict)

        logger.debug("Loaded parameters from file, which are \"" + str(parameters) + "\".")

    return(parameters)
