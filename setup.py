from setuptools import setup, find_packages
import dfpipe

setup(
    name='dfpipe',
    version=dfpipe.__version__,
    author='dfpipe',
    author_email='dfpipe2024@gmail.com',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dfpipe/dfpipe-python',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
) 