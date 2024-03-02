from setuptools import setup, find_packages

import re

VERSIONFILE = "mlx_nn/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if l and not l.startswith('#')]

setup(
    name='mlx-nn',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Bute force K-Nearest Neighbor search for Apple silicon.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/schlegelp/mlx-nn',
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='ARM GPU nearest-neighbor',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    install_requires=requirements,
    extras_require={'extras': []},
    python_requires='>=3.8',
    zip_safe=False,

    include_package_data=True

)