#!/bin/zsh
if [ $1 = "develop" ]; then
    # develop mode
    python setup.py develop
    echo "Develop mode"
elif [ $1 = "release" ]; then
    # build whl
    python -m build --wheel
    echo "Release mode"
elif [ $1 = "testpypi" ]; then
    # upload to testpypi
    twine upload -r testpypi dist/dmt_learn-$2-py3-none-any.whl --verbose
    echo "Testpypi mode"
elif [ $1 = "pypi" ]; then
    # upload to pypi
    twine upload dist/dmt_learn-$2-py3-none-any.whl --verbose
    echo "Pypi mode"
else
    echo "Invalid argument"
fi

# fi
# twine upload dist/dmt_learn-$1-py3-none-any.whl --verbose
# twine upload -r testpypi dist/dmt_learn-$1-py3-none-any.whl --verbose