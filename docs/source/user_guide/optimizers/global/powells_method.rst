===============
Powell's Method
===============

Powell's Method optimizes each dimension sequentially, performing 1D optimization
along each axis before moving to the next. This works particularly well for
**separable functions** where dimensions are independent.


Algorithm
---------

At each iteration:

1. Select a dimension to optimize
2. Perform 1D optimization along that dimension (keeping others fixed)
3. Move to the next dimension
4. Repeat through all dimensions

This sequential approach is efficient when dimensions are independent but
may struggle with coupled parameters.


When to Use
-----------

**Good for:**

- Separable objective functions (dimensions are independent)
- Functions where 1D optimization is cheap
- Low to moderate dimensional spaces

**Not ideal for:**

- Strongly coupled parameters (e.g., ``score = (x + y)^2``)
- Very high dimensions
- Functions with complex interactions between parameters


Example
-------

.. code-block:: python

    import numpy as np
    from gradient_free_optimizers import PowellsMethod

    # Separable function - works well with Powell's Method
    def separable(para):
        return -(para["x"]**2 + para["y"]**2 + para["z"]**2)

    search_space = {
        "x": np.linspace(-10, 10, 100),
        "y": np.linspace(-10, 10, 100),
        "z": np.linspace(-10, 10, 100),
    }

    opt = PowellsMethod(search_space)
    opt.search(separable, n_iter=300)

    print(f"Best: {opt.best_para}")
    print(f"Score: {opt.best_score}")


Separable vs. Non-Separable Functions
-------------------------------------

.. code-block:: python

    # Separable: each dimension can be optimized independently
    def separable(para):
        return -(para["x"]**2 + para["y"]**2)

    # Non-separable: dimensions are coupled
    def coupled(para):
        return -((para["x"] + para["y"])**2)

For non-separable functions, consider :doc:`pattern_search` or
:doc:`../local/downhill_simplex` instead.


Related Algorithms
------------------

- :doc:`pattern_search` - Structured exploration in multiple directions
- :doc:`../local/downhill_simplex` - Multi-dimensional geometric approach
- :doc:`../local/hill_climbing` - Random neighborhood exploration
