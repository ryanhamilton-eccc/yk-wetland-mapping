from setuptools import setup, find_packages

setup(
    name='ykwmp',
    version='0.1.0',
    description="Yukon Wetland Mapping Project",
    author='Ryan Hamilton',
    author_email='ryan.hamilton@ec.gc.ca',
    packages=find_packages(include=['ykwmp', 'ykwmp.*']),
)