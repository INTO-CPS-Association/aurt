import time
import unittest
from aurt.tests import logger

class TimedTest(unittest.TestCase):
    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        t = time.time() - self.start_time
        logger.debug('%s: %.3fs' % (self.id(), t))

