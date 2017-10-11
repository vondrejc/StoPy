#!/usr/bin/python

if __name__ == "__main__":
    import unittest
    from uq.unittests import Test as uq_Test
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(uq_Test))

    runner=unittest.TextTestRunner()
    runner.run(suite)
