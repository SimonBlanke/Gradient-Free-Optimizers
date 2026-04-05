from dataclasses import dataclass, field


@dataclass
class Result:
    """Internal result container used throughout the search pipeline."""

    score: float
    metrics: dict


@dataclass
class ObjectiveResult:
    """Explicit return type for objective functions.

    Avoids the ambiguity of bare tuples, which becomes critical when the
    return value needs to carry structured data alongside the score (e.g.,
    custom metrics). A plain ``(float, dict)`` tuple works today but
    collides with future multi-objective returns like ``(float, float)``.

    Parameters
    ----------
    score : float
        The objective function score.
    metrics : dict, optional
        Custom metrics to record alongside the score.

    Examples
    --------
    >>> def objective(params):
    ...     loss = params["x"] ** 2
    ...     return ObjectiveResult(score=-loss, metrics={"raw_loss": loss})
    """

    score: float
    metrics: dict = field(default_factory=dict)


def unpack_objective_result(raw) -> tuple[float, dict]:
    """Extract score and metrics from an objective function's return value.

    Single entry point for parsing objective output. All code paths that
    receive raw objective returns (serial adapter, distributed unpacking,
    minimization wrapper) should call this instead of doing their own
    isinstance checks.

    Supported return conventions (checked in this order):

    1. ``ObjectiveResult`` instance (preferred, unambiguous)
    2. ``(float, dict)`` tuple (legacy convention)
    3. ``float`` (score only, no metrics)

    Parameters
    ----------
    raw : float or tuple or ObjectiveResult
        Raw return value from an objective function.

    Returns
    -------
    tuple[float, dict]
        The (score, metrics) pair.
    """
    if isinstance(raw, ObjectiveResult):
        return raw.score, raw.metrics
    if isinstance(raw, tuple):
        return raw[0], raw[1]
    return float(raw), {}
