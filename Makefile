install-requirements:
	python -m pip install -r ./requirements/requirements.in

install-test-requirements:
	python -m pip install -r ./requirements/requirements-test.in

install-build-requirements:
	python -m pip install -r ./requirements/requirements-build.in

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

test-debug:
	cd tests; \
		python _test_debug.py

test-timings:
	python -m pytest -x -p no:warnings -rfEX tests/_test_memory.py
	python -m pytest -x -p no:warnings -rfEX tests/test_optimizers/_test_max_time.py
	python -m pytest -x -p no:warnings -rfEX tests/test_optimizers/_test_memory_warm_start.py

test-performance:
	python -m pytest -x -p no:warnings -rfEX tests/local_test_performance/local_test_local_opt.py
	python -m pytest -x -p no:warnings -rfEX tests/local_test_performance/local_test_global_opt.py
	python -m pytest -x -p no:warnings -rfEX tests/local_test_performance/local_test_grid_search.py
	python -m pytest -x -p no:warnings -rfEX tests/local_test_performance/local_test_pop_opt.py
	python -m pytest -x -p no:warnings -rfEX tests/local_test_performance/local_test_smb_opt.py

test:
	make test-gfo
	make test-timings
	make test-examples
	make test-hyper

