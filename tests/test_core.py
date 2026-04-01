from pimakerlibrary.core import make_pi


def test_make_pi_returns_float_close_to_pi():
    result = make_pi()
    assert isinstance(result, float)
    assert abs(result - 3.141592653589793) < 1e-12