from setuptools import setup

setup(
    name='sbvar',
    packages=['sbvar'],
    version="1.0.0",
    author="Rachel Ng",
    author_email="rachelng323@gmail.com",
    url="https://github.com/racng/SBvar",
    description="Varying parameter analysis for SBML models",
    license="MIT", 
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'tellurium',
        'numpy',
        'pandas',
        'matplotlib',
        'anndata'
    ]
)