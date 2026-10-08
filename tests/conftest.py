# shared fixtures for every test in this folder
import pytest
import torch

# fixtures that build records live in a .pyn module; pytest only loads
# conftest.py, so they're imported here to reach every test file
from unet_fixtures import awake, cfg, data, model, train_cfg  # noqa: F401


@pytest.fixture(autouse=True)
def _seed():
    # same RNG state per test, so random batches and inits are reproducible
    torch.manual_seed(0)
