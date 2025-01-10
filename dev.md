# Development Guide

## Code style

We follow [Google Style Python Docstring](https://google.github.io/styleguide/pyguide.html) for development.

Following is an typical example:

```python
class SampleClass:
    """Summary of class here.

    Longer class information...
    Longer class information...

    """

    def __init__(self, likes_spam: bool = False):
        """Initializes the instance based on spam preference.

        Args:
          likes_spam: Defines if instance exhibits this preference.
        """
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self, a, b):
        """Performs operation blah.

        Long description here.

        Args:
            a (int): xxx
            b (int/str): xxx

        Returns:
            t (bool): xxx
            k (int): xxx
        """
        # function implementation goes here
```

## Run unit tests

We use `tox` to run unit tests. You should install `tox` in your development environemnt
```
pip install tox
```
Currently we only use python3.10 to run tests. If you don't have python3.10 in your system, you can use conda. After conda is installed, you should install tox conda plugin by running
```
pip install tox-conda
```
After tox is ready, you can run all the unit test by running
```
tox
```
Please note tox will reuse the same virtual environment which is initialized by installing all packages listed in `requirements.txt` and `requirements-dev.txt`. If any of above files are modified, you should re-create virtual environment by running
```
tox -r
```

To run a single unit test task during development, you can run

```
pytest tests/your_test_file.py
```

### Unit tests in AzureDevops pipeline

We use AzureDevops to run unit tests before you can merge your PR to main branch. You can find the pipeline definition in `azure-pipelines.yml`.

Please note that in AzureDevops pipeline agent, no gpu is available. So you must make sure your unit tests can run on cpu to pass the CI. Two options are available:
1. Use `@replace_all_device_with('cpu')` decorator to replace all devices with cpu. Please refer to other tests for example.
2. Mark your test case only work on gpu by using `@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')` decorator. Please refer to existing tests for example.

Before you push your code, please run tests at least on GPU machines to make sure all tests can pass. GPU test cases can't be run in AzureDevops pipeline. Of course, it would be better if you can run all tests on both GPU and CPU machines.

### Run unit tests in vscode

VS Code has a great support to unit tests. You can run/debug every tests easily in VS Code. Please refer to this document to set up your environment https://code.visualstudio.com/docs/python/testing

Another trick is, if you want to step into pakcage source code, you can add the following config to your .vscode/launch.json:
```
{
    "name": "Debug Unit Test",
    "type": "python",
    "request": "test",
    "justMyCode": false,
},
```

### Write Unit Tests
1. If you need to use torchrun, please refer to `unit_test/launch_torchrun.py`, and you can find examples in `unit_tests/runtime/test_runtime_collectives.py`. Please note that `torchrun` is very slow, you should reduce its usage as possible.
2. If you want to mock up any functions/methods, please use pytest-mock.
3. **NOTE**: The name of test files and test functions must start with `test_`