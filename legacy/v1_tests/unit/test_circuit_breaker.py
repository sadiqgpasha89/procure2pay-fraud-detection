"""Unit tests for Circuit Breaker pattern and fraud fallback penalty (Section 6 & Edge Case 8)."""
import time
import pytest
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException, CircuitState


@pytest.mark.unit
def test_circuit_breaker_normal_operation():
    """Circuit breaker should remain CLOSED when calls succeed."""
    cb = CircuitBreaker("test_api", failure_threshold=3, recovery_timeout_sec=1.0)
    res = cb.call(lambda x: x * 2, 5)
    assert res == 10
    assert cb.state == CircuitState.CLOSED
    assert cb.get_risk_penalty() == 0.0


@pytest.mark.unit
def test_circuit_breaker_trips_to_open_and_falls_back():
    """Circuit breaker should trip to OPEN after threshold failures and return fallback with penalty."""
    cb = CircuitBreaker("flaky_api", failure_threshold=3, recovery_timeout_sec=0.5, fallback_penalty=0.15)

    def failing_api():
        raise ConnectionError("External API offline")

    fallback_called = False
    def fallback_fn():
        nonlocal fallback_called
        fallback_called = True
        return {"cached_status": "ACTIVE"}

    # 3 failures to trip the circuit
    for _ in range(3):
        res = cb.call(failing_api, fallback=fallback_fn)
        assert res["cached_status"] == "ACTIVE"

    assert cb.state == CircuitState.OPEN
    assert cb.get_risk_penalty() == 0.15
    assert fallback_called is True


@pytest.mark.unit
def test_circuit_breaker_half_open_recovery():
    """Circuit breaker should transition to HALF_OPEN after timeout and recover to CLOSED upon success."""
    cb = CircuitBreaker("recovery_api", failure_threshold=2, recovery_timeout_sec=0.1)

    def failing():
        raise TimeoutError("timeout")

    for _ in range(2):
        cb.call(failing, fallback=lambda: None)
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.15)

    # Next call should attempt recovery
    res = cb.call(lambda: "recovered")
    assert res == "recovered"
    assert cb.state == CircuitState.CLOSED
    assert cb.get_risk_penalty() == 0.0
