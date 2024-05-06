import setuptools

with open("requirements.txt") as f:
    install_requires = [
        line.split("#")[0].strip() for line in f if not line.startswith("#") and line.split("#")[0].strip() != ""
    ]


setuptools.setup(
  name=             'tessel',
  version=          '0.3',
  author=           'Zhiqi Lin',
  author_email=     'zhiqi.0@outlook.com',
  description=      'Schedule plan searching for composing micro-batch executions',
  long_description= '',
  packages=         ['tessel'],
  python_requires=  '>=3.8',
  install_requires= install_requires,
)
