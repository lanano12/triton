import pytest

import triton._internal_testing as internal_testing
from triton.backends.compiler import GPUTarget


@pytest.mark.parametrize(
    "arch, warp_size, expected",
    [
        ("gfx90a", 64, 64 * 1024),
        ("gfx942", 64, 64 * 1024),
        ("gfx950", 64, 160 * 1024),
        ("gfx1100", 32, 64 * 1024),
        ("gfx1151", 32, 64 * 1024),
        ("gfx1201", 32, 64 * 1024),
        ("gfx1250", 32, 320 * 1024),
    ],
)
def test_get_hip_lds_size_matches_target_features(monkeypatch, arch, warp_size, expected):
    # get_hip_lds_size() must agree with TargetFeatures::getSharedMemorySize,
    # which the compiler uses as the LDS limit.
    monkeypatch.setattr(internal_testing, "get_current_target", lambda: GPUTarget("hip", arch, warp_size))
    assert internal_testing.get_hip_lds_size() == expected
