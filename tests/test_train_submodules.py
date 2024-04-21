"""Unit tests for train submodules"""

import unittest
import doctest
from pathlib import Path

from train import load_config


# pylint: disable=C0103, W0107, W0613, W0221, C2801
def setUpModule():
    """Setup tests"""
    pass


def tearDownModule():
    """Clean after tests"""
    pass


def load_tests(loader, tests, ignore):  # Returns empty TestSuite if fails.
    """Convert DocTests from module to unittest.TestSuite."""
    tests.addTests(doctest.DocTestSuite(module=None, setUp=None, tearDown=None))
    return tests  # Returned tests run with the rest of unittests.


class TestName(unittest.TestCase):
    """Unittest."""

    maxDiff, __slots__ = None, ()

    def setUp(self):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        pass

    def tearDown(self):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass

    def addCleanup(self, function, *args, **kwargs):
        """Function called AFTER tearDown() to clean resources used on test."""
        pass

    @classmethod
    def setUpClass(cls):
        """Class method called BEFORE tests in an individual class run. """
        pass  # Probably you may not use this one. See setUp().

    @classmethod
    def tearDownClass(cls):
        """Class method called AFTER tests in an individual class run. """
        pass  # Probably you may not use this one. See tearDown().

    def test_config_file(self):
        """Run test for config file"""
        config_path = Path("templates/config.yaml.template")
        config = load_config(config_path)
        self.assertIsNotNone(config)
        self.assertEqual(config['config']['NEPTUNE_API_TOKEN'], "f2137b")


if __name__.__contains__("__main__"):
    print(__doc__)
    unittest.main()
