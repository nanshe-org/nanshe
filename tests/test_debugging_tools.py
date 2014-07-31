__author__="John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ ="$July 30, 2014 4:57:43PM$"


import StringIO
import logging

import nanshe.debugging_tools

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
        expected_result = """DEBUG:debugging_tools:Entering callable: "test".\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test".\n"""

        @nanshe.debugging_tools.log_call(self.logger)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)


    def test_log_call_2(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test".\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test".\n"""

        @nanshe.debugging_tools.log_call(self.logger, to_log_call = True)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)


    def test_log_call_3(self):
        expected_result = """"""

        @nanshe.debugging_tools.log_call(self.logger, to_log_call = False)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)


    def test_log_call_4(self):
        expected_result = """"""

        @nanshe.debugging_tools.log_call(self.logger, to_log_call = True)
        def test(a, b = 5):
            return(a + b)

        test.to_log_call = False
        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)


    def test_log_call_5(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test".\n""" + \
        """DEBUG:debugging_tools:Arguments: "(0,)\"\n""" + \
        """Keyword Arguments: "{}".\n""" + \
        """DEBUG:debugging_tools:Exiting callable: "test".\n"""

        @nanshe.debugging_tools.log_call(self.logger, to_print_args = True)
        def test(a, b = 5):
            return(a + b)

        test(0)

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)


    def test_log_call_6(self):
        expected_result = """DEBUG:debugging_tools:Entering callable: "test".\n""" + \
        """DEBUG:debugging_tools:Arguments: "('c',)\"\n""" + \
        """Keyword Arguments: "{}".\n"""

        @nanshe.debugging_tools.log_call(self.logger, to_print_args = True)
        def test(a, b = 5):
            return(a + b)

        try:
            test("c")
        except:
            pass

        self.handler.flush()
        result = self.stream.getvalue()

        print(result)

        assert(result == expected_result)

    def teardown(self):
        self.handler.close()
        self.stream.close()