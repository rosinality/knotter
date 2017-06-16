from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='knotter',
    
    version='0.2.6',
    
    description='Implementation of Mapper algorithm for Topological Data Analysis',
    long_description=long_description,
    
    author='Kim Seonghyeon',
    author_email='kim.seonghyeon@outlook.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='Topological Data Analysis,Visualization',

    packages=find_packages(),
    include_package_data=True,
    
    install_requires=[
        'aiohttp >= 0.16.5',
        'pandas >= 0.16.2',
        'numpy >= 1.9.2',
        'scipy >= 0.15.1',
        'Jinja2 >= 2.8'
    ],

    entry_points={
        'console_scripts': [
            'knotter=knotter:run_server'
        ]    
    }
)
