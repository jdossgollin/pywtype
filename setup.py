import setuptools

LONG_DESCRIPTION = """
**pywtype**: codes for weather typing

Important links
---------------
- HTML documentation: http://pywtype.readthedocs.io/en/latest
- Issue tracker: https://github.com/jdossgollin/pywtype/issues
- Source code: https://github.com/jdossgollin/pywtype
"""

setuptools.setup(
    name="pywtype",
    version="0.0.1",
    packages=setuptools.find_packages(),
    author="James Doss-Gollin",
    author_email="james.doss-gollin@columbia.edu",
    description="meteorological weather typing",
    long_description=LONG_DESCRIPTION,
    install_requires=[
        'cartopy >= 0.17',
        'netCDF4 >= 1.2',
        'matplotlib >= 3.0',
        'numpy >= 1.7',
        'pandas >= 0.15.0',
        'scipy >= 0.16',
        'scikit-learn >= 0.18',
        'xarray >= 0.10.6',
    ],
    #tests_require=['pytest >= 3.3'],
    #package_data={'aospy': ['test/data/netcdf/*.nc']},
    #scripts=['aospy/examples/aospy_main.py',
    #         'aospy/examples/example_obj_lib.py'],
    license="MIT",
    keywords="climate meteorology",
    url="https://github.com/jdossgollin/pywtype",
)
