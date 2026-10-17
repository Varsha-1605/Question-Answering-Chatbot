import asyncio

import pytest

from ai_workflow import handler
from ai_workflow.handler import retry_with_custom_backoff


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    delays = []
    monkeypatch.setattr(handler.time, "sleep", delays.append)

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(handler.asyncio, "sleep", fake_sleep)
    return delays


def flaky(failures):
    state = {"calls": 0}

    def func():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise ConnectionError("down")
        return "ok"

    return func, state


def test_retries_until_success_with_exponential_delays(no_sleep):
    func, state = flaky(2)
    assert retry_with_custom_backoff(max_retries=3, initial_delay=1, backoff_factor=2)(func)() == "ok"
    assert state["calls"] == 3
    assert no_sleep == [1, 2]


def test_raises_the_last_error_after_max_retries():
    func, state = flaky(10)
    with pytest.raises(ConnectionError):
        retry_with_custom_backoff(max_retries=3)(func)()
    assert state["calls"] == 3


def test_other_exception_types_are_not_retried():
    func, state = flaky(10)
    with pytest.raises(ConnectionError):
        retry_with_custom_backoff(max_retries=3, exception=(ValueError,))(func)()
    assert state["calls"] == 1


def test_delay_is_capped(no_sleep):
    func, _ = flaky(3)
    retry_with_custom_backoff(max_retries=4, initial_delay=10, backoff_factor=10, max_delay=15)(func)()
    assert no_sleep == [10, 15, 15]


def test_async_functions_are_retried_too():
    state = {"calls": 0}

    async def func():
        state["calls"] += 1
        if state["calls"] < 2:
            raise ConnectionError
        return "ok"

    assert asyncio.run(retry_with_custom_backoff(max_retries=3)(func)()) == "ok"
