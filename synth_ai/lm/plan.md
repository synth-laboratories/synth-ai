# Unified LM Interface Plan

## Overview
Create a unified interface that allows seamless switching between OpenAI and Synth providers for LLM-based agents. The goal is to have minimal code changes when switching providers - just different model names, provider configuration, and an optional warmup call.

## Architecture

### 1. `warmup.py` - Warmup Endpoint Wrapper
```python
# synth-ai/synth_ai/lm/warmup.py

import httpx
import asyncio
import time
from typing import Optional

async def warmup_synth_model(model_name: str, base_url: str, api_key: str, max_attempts: int = 30) -> bool:
    """Warm up a specific model on the Synth backend.
    
    Args:
        model_name: Name of the model to warm up
        base_url: Base URL of the Synth backend
        api_key: API key for authentication
        max_attempts: Maximum number of polling attempts
    
    Returns:
        True if model is successfully warmed up, False otherwise
    """
    print(f"🔥 Warming up {model_name}...")
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                response = await client.post(
                    f"{base_url}/warmup/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print(f"✅ {model_name} warmed up successfully")
                    return True
                elif response.status_code == 202:
                    print(f"⏳ {model_name} warming up... (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(2.0)
                else:
                    print(f"❌ Warmup failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"⚠️  Warmup attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0)
        
        print(f"❌ Failed to warm up {model_name} after {max_attempts} attempts")
        return False
```

### 2. `synth_dev_api.py` - OpenAI-Compatible Synth Client
```python
# synth-ai/synth_ai/lm/vendors/core/synth_dev_api.py

import httpx
from typing import List, Dict, Any, Optional

class AsyncSynthClient:
    """Minimal async OpenAI-compatible client for Synth backend."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    async def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """Create a chat completion (OpenAI-compatible interface)."""
        
        # Prepare payload (same as OpenAI)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Make request to Synth endpoint
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120.0
        )
        
        if response.status_code != 200:
            raise Exception(f"Synth API error: {response.status_code} - {response.text}")
        
        # Return the response exactly as OpenAI would
        return response.json()

class SyncSynthClient:
    """Minimal sync OpenAI-compatible client for Synth backend."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client()
    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """Create a chat completion (OpenAI-compatible interface)."""
        
        # Prepare payload (same as OpenAI)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Make request to Synth endpoint
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120.0
        )
        
        if response.status_code != 200:
            raise Exception(f"Synth API error: {response.status_code} - {response.text}")
        
        # Return the response exactly as OpenAI would
        return response.json()

# Convenience functions for OpenAI-style usage
async def create_chat_completion_async(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str,
    api_key: str,
    **kwargs
):
    """OpenAI-style async chat completion function."""
    client = AsyncSynthClient(base_url, api_key)
    return await client.chat_completions_create(model, messages, **kwargs)

def create_chat_completion_sync(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str,
    api_key: str,
    **kwargs
):
    """OpenAI-style sync chat completion function."""
    client = SyncSynthClient(base_url, api_key)
    return client.chat_completions_create(model, messages, **kwargs)
```

### 3. `test_crafter_react_agent_synth.py` - Synth Provider Integration
```python
# synth-ai/synth_ai/environments/examples/crafter_classic/agent_demos/test_crafter_react_agent_synth.py

"""
Crafter ReAct agent using Synth backend instead of OpenAI.
Minimal changes from OpenAI version - just different provider and warmup.
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List
from httpx import AsyncClient

# Import the unified LM interface
from synth_ai.lm.warmup import warmup_synth_model
from synth_ai.lm.vendors.core.synth_dev_api import SynthClient, SynthMessage, SynthTool

# Configuration
SYNTH_BASE_URL = "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run"
SYNTH_API_KEY = os.environ.get("SYNTH_API_KEY", "sk-test123")

# Model configuration
SYNTH_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Synth model name

class SynthReActAgent:
    """ReAct agent using Synth backend (minimal changes from OpenAI version)."""
    
    def __init__(self, model_name: str = SYNTH_MODEL, max_turns: int = 20, verbose: bool = False):
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.client = SynthClient(SYNTH_BASE_URL, SYNTH_API_KEY)
        self.tools = self._get_synth_tools()
    
    def _get_synth_tools(self) -> List[SynthTool]:
        """Get OpenAI-compatible tool definitions for Synth."""
        return [
            SynthTool(
                type="function",
                function={
                    "name": "interact",
                    "description": "Perform 1-5 actions in sequence in the Crafter environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of 1-5 action names to execute in sequence"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why these actions were chosen"
                            }
                        },
                        "required": ["actions", "reasoning"]
                    }
                }
            ),
            SynthTool(
                type="function",
                function={
                    "name": "terminate",
                    "description": "End the episode when finished or no progress can be made.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for termination"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            )
        ]
    
    def get_system_message(self) -> str:
        """Same system message as OpenAI version."""
        return """You are CrafterAgent playing Crafter survival environment. Your goal is to unlock as many achievements as possible while staying alive.

You will see a semantic map view showing your surroundings. Use this to navigate toward resources.

Key mechanics:
• 'do' action: collect wood from trees, stone from deposits, food from cows/plants
• 'do' does nothing on grass/water - move to find resources first
• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools
• Sleep when energy low to restore and unlock wake_up achievement
• Use semantic map view to navigate toward resources you can see

Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop

Strategy:
1. Look at the semantic map to see what's around you
2. Move toward trees to collect wood with 'do'
3. Once you have wood, place a table to enable crafting
4. Make a wood pickaxe to collect stone more efficiently
5. Progress to stone pickaxe, then iron tools
6. Eat food when health is low, sleep when energy is low

You should provide 1-5 actions in sequence for efficient gameplay. Use the semantic map view to navigate toward visible resources.

Example good action sequences:
- ['move_right', 'move_right', 'do'] (move to tree and collect wood)
- ['place_table', 'make_wood_pickaxe'] (craft progression)
- ['move_up', 'do', 'move_down', 'do'] (collect from multiple resources)

Be strategic and use the map view to find resources! Focus on unlocking achievements."""
    
    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Same observation formatting as OpenAI version."""
        # [Same implementation as OpenAI version]
        # ... (copy the format_observation method from test_crafter_react_agent_openai.py)
        pass
    
    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision using Synth backend."""
        messages = [
            SynthMessage(role="system", content=system_message),
            SynthMessage(role="user", content=f"Turn {turn + 1}/{self.max_turns}\n\n{obs}")
        ]
        
        try:
            response = await self.client.chat_completions_create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            # Extract tool calls (same logic as OpenAI version)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                return {
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments)
                }
            else:
                # Fallback to text response parsing
                content = response.choices[0].message.content
                return self._parse_text_response(content)
                
        except Exception as e:
            print(f"❌ Synth API error: {e}")
            return {"name": "interact", "parameters": {"actions": ["do"], "reasoning": "error fallback"}}
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse text response as fallback (same as OpenAI version)."""
        # [Same implementation as OpenAI version]
        # ... (copy the text parsing logic)
        pass

async def main():
    """Main function with warmup integration."""
    
    # 1. WARMUP CALL (only difference from OpenAI version)
    print(f"🔥 Warming up Synth model: {SYNTH_MODEL}")
    warmup_success = await warmup_synth_model(
        model_name=SYNTH_MODEL,
        base_url=SYNTH_BASE_URL,
        api_key=SYNTH_API_KEY
    )
    
    if not warmup_success:
        print("❌ Failed to warm up model, exiting")
        return
    
    print("✅ Model warmed up successfully!")
    
    # 2. CREATE AGENT (same as OpenAI version)
    agent = SynthReActAgent(model_name=SYNTH_MODEL, max_turns=20, verbose=True)
    
    # 3. RUN EPISODES (same logic as OpenAI version)
    # [Copy the episode running logic from test_crafter_react_agent_openai.py]
    # ... (same implementation)
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Benefits

### 1. **Minimal Code Changes**
- Only 3 differences when switching from OpenAI to Synth:
  - Different model name (`gpt-4o-mini` → `Qwen/Qwen2.5-7B-Instruct`)
  - Different provider (OpenAI client → Synth client)
  - Optional warmup call at script start

### 2. **Unified Interface**
- Same message format (`SynthMessage` compatible with OpenAI)
- Same tool calling interface
- Same response structure
- Same error handling patterns

### 3. **Easy Provider Switching**
```python
# OpenAI version
import openai
client = openai.AsyncOpenAI(api_key="sk-...")
response = await client.chat.completions.create(...)

# Synth version  
from synth_ai.lm.vendors.core.synth_dev_api import SynthClient
client = SynthClient(base_url="...", api_key="...")
response = await client.chat_completions_create(...)
```

### 4. **Warmup Integration**
```python
# Only needed for Synth, not OpenAI
await warmup_synth_model("Qwen/Qwen2.5-7B-Instruct", base_url, api_key)
```

## Implementation Steps

1. **Create `warmup.py`** - Warmup endpoint wrapper
2. **Create `synth_dev_api.py`** - OpenAI-compatible Synth client
3. **Create `test_crafter_react_agent_synth.py`** - Synth provider integration
4. **Test with existing Crafter agent** - Verify compatibility
5. **Document usage patterns** - For easy provider switching

## Usage Example

```python
# OpenAI version
python test_crafter_react_agent_openai.py --model gpt-4o-mini

# Synth version (only difference is model name and warmup)
python test_crafter_react_agent_synth.py --model Qwen/Qwen2.5-7B-Instruct
```

This approach provides a clean, unified interface that makes it trivial to switch between providers while maintaining all the sophisticated features of the ReAct agent framework.


Here is a refined version of your **Unified LM Interface Plan**, with clearer structure, streamlined language, and consistent tone for production-level documentation.

---

# Unified LM Interface Plan

## Overview

Design a unified LLM interface enabling seamless switching between OpenAI and Synth providers with minimal code changes. Differences are isolated to model names, provider configuration, and an optional warmup call.

---

## Architecture

### 1. `warmup.py`: Synth Model Warmup Wrapper

```python
# synth_ai/lm/warmup.py

import httpx, asyncio
from typing import Optional

async def warmup_synth_model(model_name: str, base_url: str, api_key: str, max_attempts: int = 30) -> bool:
    """Poll Synth warmup endpoint until model is ready."""
    print(f"🔥 Warming up {model_name}...")

    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                resp = await client.post(
                    f"{base_url}/warmup/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0,
                )

                if resp.status_code == 200:
                    print(f"✅ {model_name} warmed up successfully")
                    return True
                elif resp.status_code == 202:
                    print(f"⏳ Warming up... (attempt {attempt+1}/{max_attempts})")
                    await asyncio.sleep(2.0)
                else:
                    print(f"❌ Warmup failed: {resp.status_code}")
                    return False

            except Exception as e:
                print(f"⚠️  Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1.0)

    print(f"❌ Failed to warm up {model_name} after {max_attempts} attempts")
    return False
```

---

### 2. `synth_dev_api.py`: OpenAI-Compatible Synth Client

```python
# synth_ai/lm/vendors/core/synth_dev_api.py

import httpx
from typing import List, Dict, Any, Optional

class AsyncSynthClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    async def chat_completions_create(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if v is not None}
        }

        resp = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120.0,
        )

        if resp.status_code != 200:
            raise Exception(f"Synth API error: {resp.status_code} - {resp.text}")
        return resp.json()

class SyncSynthClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client()

    def chat_completions_create(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if v is not None}
        }

        resp = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120.0,
        )

        if resp.status_code != 200:
            raise Exception(f"Synth API error: {resp.status_code} - {resp.text}")
        return resp.json()

# Convenience wrappers
async def create_chat_completion_async(model, messages, base_url, api_key, **kwargs):
    return await AsyncSynthClient(base_url, api_key).chat_completions_create(model, messages, **kwargs)

def create_chat_completion_sync(model, messages, base_url, api_key, **kwargs):
    return SyncSynthClient(base_url, api_key).chat_completions_create(model, messages, **kwargs)
```

---

### 3. `test_crafter_react_agent_synth.py`: Crafter Agent (Synth Version)

```python
# synth_ai/environments/examples/crafter_classic/agent_demos/test_crafter_react_agent_synth.py

"""
Crafter ReAct agent using Synth backend (drop-in replacement for OpenAI version).
"""

import asyncio, os, json
from typing import Dict, Any, List

from synth_ai.lm.warmup import warmup_synth_model
from synth_ai.lm.vendors.core.synth_dev_api import SynthClient, SynthMessage, SynthTool

SYNTH_BASE_URL = "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run"
SYNTH_API_KEY = os.environ.get("SYNTH_API_KEY", "sk-test123")
SYNTH_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class SynthReActAgent:
    def __init__(self, model_name=SYNTH_MODEL, max_turns=20, verbose=False):
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.client = SynthClient(SYNTH_BASE_URL, SYNTH_API_KEY)
        self.tools = self._tool_definitions()

    def _tool_definitions(self) -> List[SynthTool]:
        return [
            SynthTool(
                type="function",
                function={
                    "name": "interact",
                    "description": "Perform 1–5 actions in Crafter.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {"type": "array", "items": {"type": "string"}, "description": "Action list"},
                            "reasoning": {"type": "string", "description": "Rationale"}
                        },
                        "required": ["actions", "reasoning"]
                    }
                }
            ),
            SynthTool(
                type="function",
                function={
                    "name": "terminate",
                    "description": "End the episode with reason.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Termination reason"}
                        },
                        "required": ["reason"]
                    }
                }
            )
        ]

    def get_system_message(self) -> str:
        return """You are CrafterAgent playing a survival game... [truncated for brevity]"""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        # [Same as OpenAI version]
        pass

    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        messages = [
            SynthMessage(role="system", content=system_message),
            SynthMessage(role="user", content=f"Turn {turn+1}/{self.max_turns}\n\n{obs}")
        ]

        try:
            response = await self.client.chat_completions_create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0,
            )

            tool_calls = response["choices"][0]["message"].get("tool_calls")
            if tool_calls:
                return {
                    "name": tool_calls[0]["function"]["name"],
                    "parameters": json.loads(tool_calls[0]["function"]["arguments"])
                }
            return self._parse_text_response(response["choices"][0]["message"]["content"])

        except Exception as e:
            print(f"❌ Synth API error: {e}")
            return {"name": "interact", "parameters": {"actions": ["do"], "reasoning": "fallback"}}

    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        # [Same as OpenAI version]
        pass

async def main():
    print(f"🔥 Warming up Synth model: {SYNTH_MODEL}")
    if not await warmup_synth_model(SYNTH_MODEL, SYNTH_BASE_URL, SYNTH_API_KEY):
        print("❌ Warmup failed, exiting")
        return

    agent = SynthReActAgent()
    # [Run episodes here...]
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Key Benefits

### ✅ Minimal Code Delta

* Only 3 changes to switch providers:

  * Model name (`gpt-4o-mini` → `Qwen/Qwen2.5-7B-Instruct`)
  * Client class (`OpenAI` → `SynthClient`)
  * Optional warmup

### 🔁 Interchangeable Interface

* Identical message structure
* Compatible tool-calling API
* Uniform error handling and output parsing

### ⚙️ Warmup Support

```python
# Synth only (optional)
await warmup_synth_model("Qwen/Qwen2.5-7B-Instruct", base_url, api_key)
```

### 🧪 Usage Comparison

```python
# OpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key="sk-...")
response = await client.chat.completions.create(...)

# Synth
from synth_ai.lm.vendors.core.synth_dev_api import SynthClient
client = SynthClient(base_url="...", api_key="...")
response = await client.chat_completions_create(...)
```

---

## Implementation Steps

1. ✅ Add `warmup.py`
2. ✅ Implement `synth_dev_api.py` client
3. ✅ Add Synth-based ReAct agent
4. 🔬 Verify behavior parity with OpenAI
5. 📝 Document usage for both providers

---

## CLI Example

```bash
# OpenAI
python test_crafter_react_agent_openai.py --model gpt-4o-mini

# Synth (warmup + model swap)
python test_crafter_react_agent_synth.py --model Qwen/Qwen2.5-7B-Instruct
```

---

Let me know if you want a `ProviderAdapter` class for runtime switching or unified logging/stats across providers.
