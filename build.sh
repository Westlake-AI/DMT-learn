#!/bin/bash
# develop mode
# python setup.py develop

# build whl
# python -m build --wheel

# upload to testpypi
twine upload -r testpypi dist/dmtev_learn-$1-py3-none-any.whl --verbose