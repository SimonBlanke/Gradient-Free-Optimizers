dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y derivative_free_optimizers
	rm -fr build dist derivative_free_optimizers.egg-info
	python setup.py bdist_wheel
	pip install dist/*
