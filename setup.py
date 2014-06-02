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
    'package_data': {
        'pyfilterbank': [
            'sosfilt.c',
            'sosfilt64.dll',
            'sosfilt32.dll',
            'sosfilt.so'
        ]
    }
}
setup(**settings)
