from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import sys, pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='mmb_perceptron',
    version='0.0.1',
    description='Perceptron algorithms for machine learning',
    url='http://github.com/mbollmann/perceptron/',
    license='MIT License',
    author='Marcel Bollmann',
    author_email='bollmann@linguistics.rub.de',
    packages=['mmb_perceptron'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    scripts=['bin/perceptron-tagger.py']
)
