"""Resilience4j-style Circuit Breaker with graceful degradation and fraud fallback penalties."""
import time
from enum import Enum
from typing import Any, Callable, Optional
import structlog

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operational state
    OPEN = "OPEN"            # Tripped due to failure threshold
    HALF_OPEN = "HALF_OPEN"  # Testing recovery after timeout


class CircuitBreakerOpenException(Exception):
    """Raised when a call is rejected because the circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """Production circuit breaker protecting third-party APIs (Sanctions, Bank Routing, Registries)."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        fallback_penalty: float = 0.15,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.fallback_penalty = fallback_penalty

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change_time: float = time.time()
        self.degradation_count = 0

    def __call__(self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs) -> Any:
        return self.call(func, *args, fallback=fallback, **kwargs)

    def call(self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs) -> Any:
        """Execute protected function with circuit state management and fallback."""
        now = time.time()

        # Check if OPEN state has expired and should transition to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout_sec):
                logger.info("circuit_breaker.transition_half_open", name=self.name)
                self.state = CircuitState.HALF_OPEN
            else:
                self.degradation_count += 1
                logger.warn(
                    "circuit_breaker.open_degradation",
                    name=self.name,
                    fallback_penalty=self.fallback_penalty
                )
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN. External dependency unavailable."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            if fallback:
                self.degradation_count += 1
                return fallback(*args, **kwargs)
            raise exc

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("circuit_breaker.recovered_closed", name=self.name)
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        """Handle execution failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.error(
            "circuit_breaker.call_failed",
            name=self.name,
            failure_count=self.failure_count,
            error=str(exc)
        )

        if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    "circuit_breaker.tripped_open",
                    name=self.name,
                    threshold=self.failure_threshold,
                    recovery_timeout_sec=self.recovery_timeout_sec
                )

    def get_risk_penalty(self) -> float:
        """Return fallback penalty score if circuit is degraded, else 0.0."""
        return self.fallback_penalty if self.state != CircuitState.CLOSED else 0.0

    def reset(self) -> None:
        """Force reset circuit breaker to normal CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
