class SearchParams(dict):
    """Dict subclass that carries optimization metadata as private attributes.

    Behaves exactly like a regular dict for the user's objective function.
    The dashboard decorator (or any other tooling) can read the metadata
    via getattr(params, '_optimizer', None) without affecting normal usage.
    """

    _optimizer = None
    _n_iter = None
    _iteration = None
    _search_space = None
    _phase = None
    _best_score = None
    _best_params = None
