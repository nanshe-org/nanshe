__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$July 30, 2014 4:57:43PM$"


import StringIO
import logging
import re

import nanshe.nanshe.debugging_tools

class TestDebuggingTools(object):
    def setup(self):
        self.stream = StringIO.StringIO()

        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))

        self.logger = logging.getLogger("debugging_tools")
        self.logger.setLevel(logging.DEBUG)

        # Otherwise, we get errors writing to closed objects.
        for each_handler in self.logger.handlers:
            self.logger.removeHandler(each_handler)

        self.logger.addHandler(self.handler)


    def test_log_call_1(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "test" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (re.match(expected_result, result).group() == result)


    def test_log_call_2(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "test" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger, to_log_call = True)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (re.match(expected_result, result).group() == result)


    def test_log_call_3(self):
        expected_result = """"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger, to_log_call = False)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (result == expected_result)


    def test_log_call_4(self):
        expected_result = """"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger, to_log_call = True)
        def test(a, b = 5):
            return(a + b)

        test.to_log_call = False
        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (result == expected_result)


    def test_log_call_5(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Arguments: "\(0,\)\"\n""" + \
        """Keyword Arguments: "\{\}"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "test" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger, to_print_args = True)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (re.match(expected_result, result).group() == result)


    def test_log_call_6(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test".\n""" + \
        """DEBUG:debugging_tools:Arguments: "('c',)\"\n""" + \
        """Keyword Arguments: "{}".\n"""

        @nanshe.nanshe.debugging_tools.log_call(self.logger, to_print_args = True)
        def test(a, b = 5):
            return(a + b)

        try:
            test("c")
        except:
            pass

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert (result == expected_result)


    def test_log_class_1(self):
        expected_result_1 = """DEBUG:debugging_tools:Entering callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "__init__" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        expected_result_2 = """DEBUG:debugging_tools:Entering callable: "__call__"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "__call__"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "__call__" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        @nanshe.nanshe.debugging_tools.log_class(self.logger)
        class Test(object):
            def __init__(self, a, b = 5):
                self.a = a
                self.b = b

            def __call__(self):
                return(self.a + self.b)

        op = Test(0)

        self.handler.flush()
        result_1 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_1)

        assert (re.match(expected_result_1, result_1).group() == result_1)

        op()

        self.handler.flush()
        result_2 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_2)

        assert (re.match(expected_result_2, result_2).group() == result_2)


    def test_log_class_2(self):
        expected_result_1 = """DEBUG:debugging_tools:Entering callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "__init__" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        expected_result_2 = "DEBUG:debugging_tools:Entering callable: \"__call__\"\.\n" + \
            "DEBUG:debugging_tools:Arguments: \"\(<debugging_tools\.Test object at 0x[0-9a-f]+>,\)\"\n" + \
            "Keyword Arguments: \"\{\}\"\.\n" + \
            "DEBUG:debugging_tools:Exiting callable: \"__call__\"\.\n" + \
            "DEBUG:debugging_tools:Run time for callable: \"__call__\" is \"[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s\"\.\n"

        @nanshe.nanshe.debugging_tools.log_class(self.logger)
        class Test(object):
            def __init__(self, a, b = 5):
                self.a = a
                self.b = b

            def __call__(self):
                return(self.a + self.b)

        op = Test(0)

        self.handler.flush()
        result_1 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_1)

        assert (re.match(expected_result_1, result_1).group() == result_1)

        Test.__call__.__dict__["to_print_args"] = True

        op()

        self.handler.flush()
        result_2 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_2)

        assert (re.match(expected_result_2, result_2).group() == result_2)


    def test_log_class_2(self):
        expected_result_1 = """DEBUG:debugging_tools:Entering callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "__init__"\.\n""" + \
        """DEBUG:debugging_tools:Run time for callable: "__init__" is "[0-9]+\.[0-9]+(e[\+\-]{1}[0-9]+)? s"\.\n"""

        expected_result_2 = """"""

        @nanshe.nanshe.debugging_tools.log_class(self.logger)
        class Test(object):
            def __init__(self, a, b = 5):
                self.a = a
                self.b = b

            def __call__(self):
                return(self.a + self.b)

        op = Test(0)

        self.handler.flush()
        result_1 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_1)

        assert (re.match(expected_result_1, result_1).group() == result_1)

        Test.__call__.__dict__["to_log_call"] = False

        op()

        self.handler.flush()
        result_2 = self.stream.getvalue()
        self.stream.truncate(0)

        print(result_2)

        assert (result_2 == expected_result_2)


    def teardown(self):
        self.handler.close()
        self.stream.close()
