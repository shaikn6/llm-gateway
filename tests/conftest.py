
import pytest


@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "Hello"}]
