# -- coding: utf-8 --

from setuptools import setup

settings = {
    'name': 'pyfilterbank',
    'version': '0.0.0',
    'description': 'Filterbanks and filtering for the acoustician and audiologists in python.',
    'url': 'http://github.com/SiggiGue/pyfilterbank',
    'author': u'Siegfried Gündert',
    'author_email': 'siefried.guendert@gmail.com',
    'license': 'MIT',
    'packages': ['pyfilterbank'],
    'zip_safe': False,
    'install_requires': ["cffi>=1.0.0"],
    'setup_requires': ["cffi>=1.0.0"],
    'cffi_modules': ['build.py:ffi'],
}
setup(**settings)
