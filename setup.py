from setuptools import setup, find_packages
import os

# Read the README file if it exists
def read_readme():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "A package for analyzing barometer and rotation/seismometer data"

setup(
    name="baroseis",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'obspy>=1.3.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.60.0',
        'multitaper>=0.1.0'
    ],
    extras_require={
        'dev': [
            'jupyter',
            'ipykernel',
            'notebook'
        ],
        'plotting': [
            'seaborn>=0.11.0'
        ]
    },
    author="Andreas Brotzer",
    author_email="andreas.brotzer@lmu.de",
    description="A comprehensive package for analyzing barometer and rotation/seismometer data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/andbro/baroseis",
    project_urls={
        "Bug Reports": "https://github.com/andbro/baroseis/issues",
        "Source": "https://github.com/andbro/baroseis",
        "Documentation": "https://github.com/andbro/baroseis/blob/main/docs/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    keywords="seismology, barometer, rotation, tilt, atmospheric pressure, geophysics",
    include_package_data=True,
    package_data={
        'baroseis': ['data/*', 'metadata/*', 'config/*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'baroseis=baroseis.cli:main',
        ],
    },
)
