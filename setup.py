from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name='fedpredict',
    # version='{{VERSION_PLACEHOLDER}}',
    version='0.0.0.1',
    packages=['fedpredict', 'fedpredict.utils', 'fedpredict.utils.compression_methods'],
    url='https://github.com/claudiocapanema/fedpredict',
    license='',
    author='claudio',
    author_email='claudiogs.capanema@gmail.com',
    description='FedPredict is a personalization plugin for Federated Learning methods'
)
