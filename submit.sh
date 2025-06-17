python3 -m pip install build twine
python3 -m build
twine check dist/*
twine upload dist/*