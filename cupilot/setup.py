import setuptools

with open("requirements.txt") as f:
    install_requires = [
        line.split("#")[0].strip() for line in f if not line.startswith("#") and line.split("#")[0].strip() != ""  
    ]

setuptools.setup(
    name=             'cupilot',
    version=          '0.1',
    author=           'Zhiqi Lin',
    description=      'Automated distributed DNN execution policy with expert knowledges',
    packages=         ['cupilot'],
    python_requires=  '>=3.8',
    install_requires= install_requires,
)