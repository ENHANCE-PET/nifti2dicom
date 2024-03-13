from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='nifti2dicom',
    version='1.1.5',
    packages=find_packages(),
    install_requires=[
        'pydicom',
        'nibabel',
        'numpy',
        'emoji',
        'rich',
        'pydicom',
        'highdicom',
        'pyfiglet'
    ],
    entry_points={
        'console_scripts': [
            'nifti2dicom=nifti2dicom.converter:main',
        ],
    },
    author='Lalith Kumar Shiyam Sundar',
    description='A package to convert NIfTI images to DICOM format using a reference DICOM series.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='nifti dicom converter',
)
