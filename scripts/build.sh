# Build the distribution files
python -m build

# Upload to TestPyPI (optional but recommended)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*