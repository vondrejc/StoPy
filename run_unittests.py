#!/usr/bin/python
import unittest
import os
from general import Timer


class Test_main(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tutorials(self): # test tutorials
        dirs = ['research_articles/SCE/']
        for dr in dirs:
            for filen in os.listdir(dr):
                pth = os.path.join(dr, filen)
                if os.path.isfile(pth) and filen[0] not in ['_']:
                    print('running "{}"...'.format(pth))
                    tic = Timer(name='run file')
                    exec(compile(open(pth).read(), filen, 'exec'), {'__name__': 'test'})
                    tic.measure(print_time=True)
                    print('...done')


if __name__ == "__main__":
    from uq.unittests import Test as uq_Test
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_main))
    suite.addTest(unittest.makeSuite(uq_Test))

    runner=unittest.TextTestRunner()
    runner.run(suite)
