import pytest

from rl_fun.fourier.spec import format_error


@pytest.mark.parametrize(
    ("fraction", "text"),
    [(0.42, "42 %"), (0.0421, "4.2 %"), (0.0042, "0.42 %"), (0.00001, "<0.01 %"), (0.0, "<0.01 %")],
)
def test_format_error(fraction, text):
    assert format_error(fraction) == text
