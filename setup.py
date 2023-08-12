
from setuptools import setup, find_packages

setup(
    name='nifti2dicom',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[
        'pydicom',
        'nibabel',
        'numpy',
        'emoji',
        'rich'
    ],
    entry_points={
        'console_scripts': [
            'nifti2dicom=nifti2dicom.converter:main',
        ],
    },
    author='Lalith Kumar Shiyam Sundar',
    description='A package to convert NIfTI images to DICOM format using a reference DICOM series.',
    license='MIT',
    keywords='nifti dicom converter',
)
