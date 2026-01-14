.. _about_citation:

========
Citation
========

If you use Gradient-Free-Optimizers in your research or projects, please consider citing it.

----

BibTeX Entry
------------

For academic papers and publications:

.. code-block:: bibtex

    @software{gradient_free_optimizers,
        author = {Simon Blanke},
        title = {Gradient-Free-Optimizers: Simple and reliable optimization with
                 local, global, population-based and sequential techniques in
                 numerical search spaces},
        year = {2019},
        url = {https://github.com/SimonBlanke/Gradient-Free-Optimizers},
        version = {1.6.0}
    }

.. note::

   Update the ``version`` field to match the version you used in your work.
   You can check the version with:

   .. code-block:: python

       import gradient_free_optimizers
       print(gradient_free_optimizers.__version__)

----

Text Citation
-------------

For less formal contexts:

    Blanke, S. (2019). Gradient-Free-Optimizers: Simple and reliable optimization
    with local, global, population-based and sequential techniques in numerical
    search spaces. https://github.com/SimonBlanke/Gradient-Free-Optimizers

----

Citing Specific Algorithms
---------------------------

If you use specific algorithms from GFO, consider also citing the original papers:

**Bayesian Optimization**:

.. code-block:: bibtex

    @article{mockus1978bayesian,
        title={The application of Bayesian methods for seeking the extremum},
        author={Mockus, Jonas},
        journal={Towards Global Optimization},
        volume={2},
        pages={117--129},
        year={1978}
    }

**Tree-structured Parzen Estimator (TPE)**:

.. code-block:: bibtex

    @inproceedings{bergstra2011algorithms,
        title={Algorithms for hyper-parameter optimization},
        author={Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
        booktitle={Advances in Neural Information Processing Systems},
        pages={2546--2554},
        year={2011}
    }

**Particle Swarm Optimization**:

.. code-block:: bibtex

    @inproceedings{kennedy1995particle,
        title={Particle swarm optimization},
        author={Kennedy, James and Eberhart, Russell},
        booktitle={Proceedings of IEEE International Conference on Neural Networks},
        volume={4},
        pages={1942--1948},
        year={1995}
    }

**Differential Evolution**:

.. code-block:: bibtex

    @article{storn1997differential,
        title={Differential evolution--a simple and efficient heuristic for global optimization over continuous spaces},
        author={Storn, Rainer and Price, Kenneth},
        journal={Journal of Global Optimization},
        volume={11},
        number={4},
        pages={341--359},
        year={1997}
    }

----

Related Work
------------

If you're using GFO through Hyperactive, cite both:

.. code-block:: bibtex

    @software{hyperactive,
        author = {Simon Blanke},
        title = {Hyperactive: An optimization and data collection toolbox for
                 convenient and fast prototyping of computationally expensive models},
        year = {2019},
        url = {https://github.com/SimonBlanke/Hyperactive}
    }

    @software{gradient_free_optimizers,
        author = {Simon Blanke},
        title = {Gradient-Free-Optimizers},
        year = {2019},
        url = {https://github.com/SimonBlanke/Gradient-Free-Optimizers}
    }

----

DOI and Zenodo
--------------

For a permanent identifier, you can cite the Zenodo archive:

.. note::

   A Zenodo DOI may be available for specific releases. Check the
   `GitHub repository <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
   for the latest DOI badge.

----

Acknowledgments in Papers
--------------------------

If you prefer to acknowledge GFO in your paper's acknowledgments section:

    "We thank Simon Blanke for developing Gradient-Free-Optimizers, which was
    used for hyperparameter optimization in this work."

Or:

    "Optimization experiments were conducted using the Gradient-Free-Optimizers
    library (Blanke, 2019)."

----

Let Us Know
-----------

We'd love to hear about your work using GFO! If you publish a paper or project:

- Email: simon.blanke@yahoo.com
- Open a `discussion <https://github.com/SimonBlanke/Gradient-Free-Optimizers/discussions>`_
- Tag on social media

We may feature your work in future documentation or announcements.

----

Commercial Use
--------------

GFO is released under the MIT License and can be used freely in commercial
applications without citation requirements. However, citations in technical
documentation or papers are appreciated!

See :doc:`license` for details.
