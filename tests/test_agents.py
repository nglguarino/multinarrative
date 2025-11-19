"""
Tests for agent functionality.
"""

import pytest
from src.models.agents import Agent, create_agent_pool, MultiAgentConsensus


def test_agent_initialization():
    """Test that agents can be initialized with required parameters."""
    # Mock client
    class MockClient:
        pass
    
    agent = Agent(
        name="test-model",
        client=MockClient(),
        provider="openai",
        temperature=0.1,
        max_tokens=100
    )
    
    assert agent.name == "test-model"
    assert agent.provider == "openai"
    assert agent.temperature == 0.1
    assert agent.max_tokens == 100


def test_narrative_parsing():
    """Test narrative parsing from agent responses."""
    consensus = MultiAgentConsensus([])
    
    # Test numbered list
    response = """1. First narrative
2. Second narrative
3. Third narrative"""
    
    narratives = consensus._parse_narratives(response)
    assert len(narratives) == 3
    assert "First narrative" in narratives
    
    # Test bullet points
    response = """- Bullet narrative one
- Bullet narrative two"""
    
    narratives = consensus._parse_narratives(response)
    assert len(narratives) == 2
    assert "Bullet narrative one" in narratives


def test_agent_pool_creation():
    """Test that agent pool can be created with config."""
    config = {
        'api_keys': {
            'openai': None,  # Will skip if None
            'gemini': None,
            'claude': None
        }
    }
    
    agents = create_agent_pool(config)
    # Should return empty list if no API keys
    assert isinstance(agents, list)


if __name__ == '__main__':
    pytest.main([__file__])
