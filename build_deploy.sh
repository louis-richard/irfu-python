#!/bin/zsh

rm -r dist ;
python3.8 setup.py sdist bdist_wheel ;
if python3.8 -m twine check dist/* ; then
  if [ "$1" = "--test" ] ; then
    python3.8 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  else
    python3.8 -m twine upload dist/* ;
  fi
fi