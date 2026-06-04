# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Strategy objects passed as parameters to optimizers.

A strategy is a configurable component handed to an optimizer to steer how
candidates are generated or filtered. It is not an optimizer itself: it has no
search loop, it is composed into one through a constructor parameter such as
``strategy=``.
"""

from .turbo import TuRBO

__all__ = ["TuRBO"]
