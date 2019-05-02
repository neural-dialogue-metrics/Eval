from setuptools import setup

__version__ = '0.0.1'

setup(
    name='EvalHelper',
    version=__version__,
    description='Evaluation Helper',
    author='cgsdfc',
    author_email='cgsdfc@126.com',
    keywords=[
        'NL', 'CL', 'MT',
        'natural language processing',
        'computational linguistics',
        'machine translation',
    ],
    packages=[
        'eval',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic'
    ],
    license='LICENCE.txt',
    long_description=open('README.md').read(),
    install_requires=[]
)
