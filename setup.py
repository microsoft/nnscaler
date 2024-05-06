import setuptools

with open("requirements.txt") as f:
    install_requires = [
        line.split("#")[0].strip() for line in f if not line.startswith("#") and line.split("#")[0].strip() != ""  
    ]

setuptools.setup(
    name=             'cube',
    version=          '0.2',
    author=           'Cube Team',
    description=      'Parallelize DNN Traning from A Systematic Way',
    long_description= 'Parallelize DNN Traning from A Systematic Way',
    packages=         ['cube'],
    python_requires=  '>=3.8',
    install_requires= install_requires,
)
