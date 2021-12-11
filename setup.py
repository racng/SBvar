from setuptools import setup

setup(
    name='sbvar',
    packages=['sbvar'],
    version="1.0.0",
    author="Rachel Ng",
    author_email="rachelng323@gmail.com",
    url="https://github.com/racng/sbvar",
    description="Varying parameter analysis for SBML models",
    license="MIT", 
    python_requires='>=3.7',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        'Operating System :: OS Independent',
    ],
    install_requires=[
        'antimony',
        'tellurium',
        'libroadrunner',
        'rrplugins',
        'numpy',
        'pandas',
        'matplotlib',
        'anndata',
        'scikit-learn',
        'pytest'
    ]
)
