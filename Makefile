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

test:
	cd tests/; \
		pytest HillClimbing.py -p no:warnings; \
	    pytest StochasticHillClimbing.py -p no:warnings; \
	    pytest TabuSearch.py -p no:warnings; \
	    pytest RandomRestartHillClimbing.py -p no:warnings; \
	    pytest RandomAnnealing.py -p no:warnings; \
	    pytest SimulatedAnnealing.py -p no:warnings; \
	    pytest StochasticTunneling.py -p no:warnings; \
	    pytest ParallelTempering.py -p no:warnings; \
	    pytest ParticleSwarm.py -p no:warnings; \
	    pytest EvolutionStrategy.py -p no:warnings; \
	    pytest Bayesian.py -p no:warnings; \
	    pytest TPE.py -p no:warnings; \
	    pytest DecisionTree.py -p no:warnings
