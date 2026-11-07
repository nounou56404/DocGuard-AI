from importlib.util import find_spec


def test_package_exists():
    assert find_spec("docguard_ai") is not None
