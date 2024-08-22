install-test-requirements:
	python -m pip install .[test]

install-build-requirements:
	python -m pip install .[build]

build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y gradient-free-optimizers
	rm -fr build dist *.egg-info

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

test-visual:
	cd tests/local; \
		python _visualize_search_paths.py

test-gfo:
	python -m pytest -x -p no:warnings -rfEX tests/

test-examples:
	cd tests; \
		python _test_examples.py

test-hyper:
	# test if new version of gfo works with current release of hyperactive 
	pip install --upgrade --force-reinstall hyperactive; \
	make install; \
	cd ../Hyperactive; \
		make test

test-timings:
	python -m pytest -x -p no:warnings -rfEX tests/_test_memory.py
	python -m pytest -x -p no:warnings -rfEX tests/test_optimizers/_test_max_time.py
	python -m pytest -x -p no:warnings -rfEX tests/test_optimizers/_test_memory_warm_start.py

test:
	make test-gfo
	make test-timings
	make test-examples
	make test-hyper

