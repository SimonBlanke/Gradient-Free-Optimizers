from __future__ import annotations

import math
import dataclasses as _dc
import copy
from dataclasses import fields, field
from typing import Any, get_origin, get_args, Sequence, Mapping

# --------------------------------------------------------------------------- helpers
_IMMUTABLE_PRIMITIVES = (int, float, bool, str, bytes, type(None))


def _looks_immutable(v: Any) -> bool:
    """Heuristic that matches dataclasses' own rules (3.12) – expanded for tuples."""
    if isinstance(v, _IMMUTABLE_PRIMITIVES):
        return True
    if isinstance(v, tuple):
        return all(_looks_immutable(x) for x in v)
    return False  # everything else we treat as potentially mutable


def _needs_factory(v: Any) -> bool:
    return not _looks_immutable(v)


def _is_instance(value: Any, typ) -> bool:
    """Simple recursive runtime type-checker; good enough for search-spaces."""
    origin = get_origin(typ)
    if origin in {list, Sequence}:
        if not isinstance(value, Sequence):
            return False
        (elem_t,) = get_args(typ) or (object,)
        return all(_is_instance(x, elem_t) for x in value)
    if origin in {dict, Mapping}:
        if not isinstance(value, Mapping):
            return False
        kt, vt = get_args(typ) or (object, object)
        return all(
            _is_instance(k, kt) and _is_instance(v, vt) for k, v in value.items()
        )
    try:
        return isinstance(value, typ)
    except TypeError:
        return True  # “Any” or typing constructs we don’t care about


# --------------------------------------------------------------------------- core
class BaseSearchSpace:
    """
    Sub-class me and declare **annotated** attributes.
    Mutable defaults are auto-wrapped in `field(default_factory=…)`, so users
    never need to import `dataclasses.field`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Convert mutable defaults  default_factory
        for name, typ in cls.__annotations__.items():
            if hasattr(cls, name):
                val = getattr(cls, name)
                if not isinstance(val, _dc.Field) and _needs_factory(val):

                    def _factory(orig=val):
                        return copy.deepcopy(orig)

                    setattr(cls, name, field(default_factory=_factory))

        # 2. Dataclass-ify (once)
        if not _dc.is_dataclass(cls):
            _dc.dataclass(slots=True, eq=False)(cls)

        # 3. Runtime validation on construction
        user_post = getattr(cls, "__post_init__", None)

        def _post_init(self, *a, **kw):
            if user_post:
                user_post(self, *a, **kw)
            for f in fields(self):
                if not _is_instance(getattr(self, f.name), f.type):
                    raise TypeError(
                        f"{cls.__name__}.{f.name} expected {f.type}, got "
                        f"{type(getattr(self, f.name))}"
                    )

        cls.__post_init__ = _post_init

        # 4. Guard later assignments, too
        def _checked_setattr(self, name, value):
            if name in cls.__annotations__:
                if not _is_instance(value, cls.__annotations__[name]):
                    raise TypeError(
                        f"cannot assign {type(value)} to field {name} "
                        f"(expects {cls.__annotations__[name]})"
                    )
            object.__setattr__(self, name, value)

        cls.__setattr__ = _checked_setattr

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict view for legacy APIs."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def _size_of(value: Any) -> int | float:
        """Return a meaningful ‘size’ for *value*."""
        # numpy / list / tuple / range / dict / set … → len()
        try:
            return len(value)  # works for most iterables
        except TypeError:
            pass

        # frozen scipy.stats distribution?
        if hasattr(value, "ppf"):
            return math.inf

        # scalar constant
        return 1

    # ---------- computed *properties* (lazy, per-instance) ------------------
    @property
    def para_names(self) -> List[str]:  # ['x', 'lr', …]
        return [f.name for f in fields(self)]

    @property
    def n_dim(self) -> int:  # 7   (for ToySpace)
        return len(self.para_names)

    @property
    def dim_types(self) -> List[type]:  # [np.ndarray, st.rv_frozen, …]
        return [f.type for f in fields(self)]

    @property
    def dim_sizes(self) -> List[int | float]:  # [200, math.inf, 8, …]
        return [self._size_of(getattr(self, f.name)) for f in fields(self)]
