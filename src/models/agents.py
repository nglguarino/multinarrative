"""
LLM Agent implementations for narrative extraction.

This module provides agent classes for different LLM providers and
utilities for managing multi-agent consensus.
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic


class Agent:
    """Base agent class for LLM interactions."""
    
    def __init__(self, name: str, client: Any, provider: str, 
                 temperature: float = 0.1, max_tokens: int = 2500):
        """
        Initialize an LLM agent.
        
        Args:
            name: Model name (e.g., 'gpt-4o')
            client: API client object
            provider: Provider type ('openai', 'google', 'anthropic')
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.name = name
        self.client = client
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the agent.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "google":
            return self._generate_google(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate response using OpenAI-compatible API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _generate_google(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate response using Google Gemini API."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Use the client which is the genai model
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text.strip()
    
    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate response using Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


def create_agent_pool(config: Optional[Dict[str, Any]] = None) -> List[Agent]:
    """
    Create a pool of agents from different LLM providers.
    
    Args:
        config: Configuration dictionary with API keys and model settings
        
    Returns:
        List of initialized Agent objects
    """
    if config is None:
        # Fallback to environment variables
        config = {
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY'),
                'gemini': os.getenv('GEMINI_API_KEY'),
                'grok': os.getenv('GROK_API_KEY'),
                'deepseek': os.getenv('DEEPSEEK_API_KEY'),
                'claude': os.getenv('CLAUDE_API_KEY')
            }
        }
    
    agents = []
    
    # GPT-4
    if config['api_keys'].get('openai'):
        gpt_client = OpenAI(api_key=config['api_keys']['openai'])
        agents.append(Agent(
            name="gpt-4o",
            client=gpt_client,
            provider="openai"
        ))
    
    # Gemini
    if config['api_keys'].get('gemini'):
        genai.configure(api_key=config['api_keys']['gemini'])
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        agents.append(Agent(
            name="gemini-2.0-flash-exp",
            client=gemini_model,
            provider="google"
        ))
    
    # Grok
    if config['api_keys'].get('grok'):
        grok_client = OpenAI(
            api_key=config['api_keys']['grok'],
            base_url="https://api.x.ai/v1"
        )
        agents.append(Agent(
            name="grok-2-1212",
            client=grok_client,
            provider="openai"
        ))
    
    # DeepSeek
    if config['api_keys'].get('deepseek'):
        deepseek_client = OpenAI(
            api_key=config['api_keys']['deepseek'],
            base_url="https://api.deepseek.com/v1"
        )
        agents.append(Agent(
            name="deepseek-chat",
            client=deepseek_client,
            provider="openai"
        ))
    
    # Claude
    if config['api_keys'].get('claude'):
        claude_client = Anthropic(api_key=config['api_keys']['claude'])
        agents.append(Agent(
            name="claude-sonnet-4-20250514",
            client=claude_client,
            provider="anthropic"
        ))
    
    return agents


class MultiAgentConsensus:
    """Manages consensus-based generation across multiple agents."""
    
    def __init__(self, agents: List[Agent]):
        """
        Initialize consensus manager.
        
        Args:
            agents: List of Agent objects
        """
        self.agents = agents
    
    def generate_with_consensus(self, prompt: str, system_prompt: Optional[str] = None,
                                min_agreement: int = 3) -> List[str]:
        """
        Generate responses from all agents and extract consensus.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            min_agreement: Minimum agents that must agree (not currently enforced)
            
        Returns:
            List of narrative strings from all agents
        """
        responses = []
        for agent in self.agents:
            try:
                response = agent.generate(prompt, system_prompt)
                # Parse narratives from response
                narratives = self._parse_narratives(response)
                responses.extend(narratives)
            except Exception as e:
                print(f"Warning: Agent {agent.name} failed: {e}")
                continue
        
        return responses
    
    def _parse_narratives(self, response: str) -> List[str]:
        """
        Parse narrative strings from agent response.
        
        Expected format: Each narrative on a new line, optionally numbered.
        
        Args:
            response: Raw agent response
            
        Returns:
            List of parsed narrative strings
        """
        narratives = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit():
                # Find the first space or period after digits
                idx = 0
                while idx < len(line) and (line[idx].isdigit() or line[idx] in ['.', ')']):
                    idx += 1
                line = line[idx:].strip()
            
            # Remove bullet points
            if line.startswith('- ') or line.startswith('• '):
                line = line[2:].strip()
            
            if line:
                narratives.append(line)
        
        return narratives
