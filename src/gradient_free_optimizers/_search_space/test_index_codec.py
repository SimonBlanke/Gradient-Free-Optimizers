import pytest  # type: ignore
import numpy as np
import random

from ._space import Space
from ._index_codec import IndexCodec


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="module")
def spec():
    spec = {
        "x": (-10.0, 10.0),  # Real
        "filters": [16, 32, 64, 128],  # Categorical
        "act": ["relu", "gelu", "tanh"],  # Categorical
        "dropout": (0.0, 1.0),  # Real
        "use_bn": [True, False],  # Bool
        "seed": 42,  # Fixed
    }
    return spec


# ---------------------------------------------------------------- tests
def test_scalar_roundtrip(spec, rng):
    codec = IndexCodec(spec)
    for _ in range(100):
        simple_space = Space(spec)
        p = simple_space.sample(rng=rng)
        idx = codec.encode(p)
        p2 = codec.decode(idx)
        # Round-trip idempotency (encode(decode()) == encode())
        assert np.array_equal(codec.encode(p2), idx)
        # All indices within bounds
        assert np.all(idx <= codec.max_index)
        assert np.all(idx >= 0)


def test_dict_interface(spec):
    codec = IndexCodec(spec)
    simple_space = Space(spec)
    d = Space(spec).sample(as_dict=True)
    idx = codec.encode(d)
    decoded = codec.decode(idx, as_dict=True)
    assert decoded.keys() == d.keys()


def test_batch_encoding(spec, rng):
    codec = IndexCodec(spec)
    simple_space = Space(spec)
    pts = simple_space.sample(16, rng=rng)
    enc = codec.encode_many(pts)
    assert enc.shape == (16, len(simple_space))
    dec = np.asarray(codec.decode_many(enc))
    assert dec.shape == (16, len(simple_space))
    # Check that encode_many / decode_many is consistent with scalar variant
    for p_row, enc_row in zip(pts, enc):
        assert np.array_equal(codec.encode(p_row), enc_row)


def test_to_dict(spec, rng):
    codec = IndexCodec(spec)
    simple_space = Space(spec)
    tup = tuple(simple_space.sample(rng=rng))  # ndarray -> tuple
    d = codec.to_dict(tup)
    assert list(d.keys()) == simple_space.names
    for k, v in zip(simple_space.names, tup):
        assert d[k] == v
    # already-mapping path
    assert codec.to_dict(d) == d


def test_error_handling(spec):
    simple_space = Space(spec)
    codec = IndexCodec(spec)
    with pytest.raises(ValueError):
        codec.encode([1, 2, 3])  # wrong length
    with pytest.raises(ValueError):
        codec.decode([0, 1])  # wrong length
    with pytest.raises(ValueError):
        codec.to_dict([1, 2])  # wrong length
