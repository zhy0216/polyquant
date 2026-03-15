from unittest.mock import patch, MagicMock
import pytest
from polyquant.utils import retry_with_backoff


@patch("polyquant.utils.time.sleep")
def test_succeeds_first_try(mock_sleep):
    @retry_with_backoff(max_retries=3)
    def succeed():
        return 42

    assert succeed() == 42
    mock_sleep.assert_not_called()


@patch("polyquant.utils.time.sleep")
def test_retries_then_succeeds(mock_sleep):
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert call_count == 3
    assert mock_sleep.call_count == 2


@patch("polyquant.utils.time.sleep")
def test_raises_after_max_retries(mock_sleep):
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def always_fail():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        always_fail()
    assert mock_sleep.call_count == 2


@patch("polyquant.utils.time.sleep")
def test_only_specified_exceptions_trigger_retry(mock_sleep):
    @retry_with_backoff(max_retries=3, exceptions=(ValueError,))
    def raise_type_error():
        raise TypeError("wrong type")

    with pytest.raises(TypeError, match="wrong type"):
        raise_type_error()
    mock_sleep.assert_not_called()
