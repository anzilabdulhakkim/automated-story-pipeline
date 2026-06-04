import pytest

from router import ModelTier, RequestRouter


@pytest.fixture
def test_router():
    return RequestRouter()

def test_route_to_imagen(test_router):
    decision = test_router.route(user_text="Generate an image of a cat")
    assert decision.tier == ModelTier.IMAGEN

def test_route_to_flash(test_router):
    decision = test_router.route(user_text="Tell me a story about a cat")
    assert decision.tier == ModelTier.FLASH

def test_force_tier(test_router):
    decision = test_router.route(user_text="Generate an image", force_tier=ModelTier.FLASH)
    assert decision.tier == ModelTier.FLASH
