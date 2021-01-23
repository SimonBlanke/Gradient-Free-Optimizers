dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y gradient_free_optimizers
	rm -fr build dist gradient_free_optimizers.egg-info
	python setup.py bdist_wheel
	pip install dist/*

test-visual:
	cd tests/local; \
		python _visualize_search_paths.py

test-examples:
	cd tests; \
		python test_examples.py

test-hyper:
	# test if new version of gfo works with current release of hyperactive 
	pip install --upgrade --force-reinstall hyperactive; \
	make install; \
	cd ../Hyperactive; \
		make test

test-debug:
	cd tests; \
		python _test_debug.py
	
test:
	python -m pytest -x -p no:warnings -rfEX tests/
	make test-hyper
	make test-examples

