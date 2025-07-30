#!/usr/bin/env python3
"""
Custom evaluation pipelines for analyzing Crafter traces using Gemini-1.5-flash.
"""

import json
import duckdb
from typing import List, Dict, Any, Optional
import asyncio
import os
import sys
import random

# Add the synth_ai path to import base classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
from synth_ai.evals.base import Judgement, BaseEval
from synth_ai.lm.core.main_v2 import LM

class MisunderstoodCrafterRulesEval(BaseEval):
    """Evaluate if the agent misunderstood Crafter game rules."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.lm = LM(
            model_name=model_name,
            formatting_model_name="gpt-4o-mini",
            temperature=0.3  # Increased temperature for more variety
        )
    
    async def run(self, session_data: Dict[str, Any]) -> List[Judgement]:
        """Analyze if the agent misunderstood Crafter rules using LLM."""
        
        # Extract relevant data from session
        actions = session_data.get("actions", [])
        invalid_actions = session_data.get("invalid_actions", [])
        achievements = session_data.get("achievements", [])
        inventory_changes = session_data.get("inventory_changes", [])
        total_steps = session_data.get("total_steps", 0)
        
        # Add some randomization to the prompt
        random_seed = random.randint(1, 1000)
        
        # Create analysis prompt with more specific instructions
        prompt = f"""
You are an expert evaluator analyzing a Crafter game session to determine if the agent misunderstood the game rules.

CRAFTER GAME RULES:
- The agent can move in 4 directions: up, down, left, right
- The agent can perform actions like: collect, craft, place, eat, sleep
- Valid actions depend on what's nearby and what's in inventory
- The goal is to achieve various crafting milestones

SESSION DATA (Seed: {random_seed}):
Total steps: {total_steps}
Actions taken: {actions[:30]}  # First 30 actions
Invalid actions: {invalid_actions[:20]}  # First 20 invalid actions
Achievements unlocked: {achievements}
Inventory changes: {inventory_changes[:10]}  # First 10 inventory changes

ANALYSIS TASK:
Analyze if the agent misunderstood Crafter rules. Look for:
1. Repeated invalid actions that suggest rule confusion
2. Actions that don't make sense given the game state
3. Missing obvious valid actions
4. Inefficient action patterns
5. Specific rule violations (movement, crafting, collection)

Provide your analysis in this EXACT JSON format (no additional text):
{{
    "score": <float 0-1, where 1=severe misunderstanding>,
    "reasoning": "<detailed explanation>",
    "evidence": ["<specific example 1>", "<specific example 2>", ...],
    "rule_violations": ["<specific rule misunderstood 1>", ...]
}}

Focus on concrete evidence from the session data. Be specific about what rules the agent seems to misunderstand.
"""
        
        try:
            # Use the existing LM infrastructure
            response = await self.lm.respond_async(
                system_message="You are an expert evaluator analyzing AI agent behavior in games. Respond only with valid JSON.",
                user_message=prompt
            )
            
            print(f"DEBUG - Raw LLM response: {response.raw_response[:200]}...")
            
            # Parse JSON response
            try:
                result = json.loads(response.raw_response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.raw_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "score": 0.5,
                        "reasoning": "Could not parse LLM response",
                        "evidence": ["Response parsing failed"],
                        "rule_violations": []
                    }
            
            return [Judgement(
                criteria="misunderstood_crafter_rules",
                score=result.get("score", 0.5),
                reasoning=result.get("reasoning", "No reasoning provided"),
                evidence=result.get("evidence", [])
            )]
            
        except Exception as e:
            return [Judgement(
                criteria="misunderstood_crafter_rules",
                score=0.5,
                reasoning=f"Evaluation failed: {str(e)}",
                evidence=[f"Error: {str(e)}"]
            )]

class WastedTimeEval(BaseEval):
    """Evaluate if the agent wasted time in inefficient actions."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.lm = LM(
            model_name=model_name,
            formatting_model_name="gpt-4o-mini",
            temperature=0.3  # Increased temperature for more variety
        )
    
    async def run(self, session_data: Dict[str, Any]) -> List[Judgement]:
        """Analyze if the agent wasted time inefficiently using LLM."""
        
        # Extract relevant data from session
        actions = session_data.get("actions", [])
        invalid_actions = session_data.get("invalid_actions", [])
        achievements = session_data.get("achievements", [])
        inventory_changes = session_data.get("inventory_changes", [])
        total_steps = session_data.get("total_steps", 0)
        
        # Add some randomization to the prompt
        random_seed = random.randint(1, 1000)
        
        # Create analysis prompt with more specific instructions
        prompt = f"""
You are an expert evaluator analyzing a Crafter game session to determine if the agent wasted time inefficiently.

EFFICIENCY CRITERIA:
- Repeated failed actions that could be avoided
- Unnecessary movement patterns
- Inefficient resource gathering
- Poor prioritization of goals
- Actions that don't contribute to achievements
- Time spent on non-productive activities

SESSION DATA (Seed: {random_seed}):
Total steps: {total_steps}
Actions taken: {actions[:40]}  # First 40 actions
Invalid actions: {invalid_actions[:25]}  # First 25 invalid actions
Achievements unlocked: {achievements}
Inventory changes: {inventory_changes[:15]}  # First 15 inventory changes

ANALYSIS TASK:
Analyze if the agent wasted time inefficiently. Look for:
1. Repeated invalid actions that waste steps
2. Inefficient movement patterns
3. Poor resource gathering strategies
4. Actions that don't advance toward goals
5. Missed opportunities for better actions
6. Time spent on non-productive activities

Provide your analysis in this EXACT JSON format (no additional text):
{{
    "score": <float 0-1, where 1=severe time wasting>,
    "reasoning": "<detailed explanation>",
    "evidence": ["<specific example 1>", "<specific example 2>", ...],
    "inefficiencies": ["<specific inefficiency 1>", ...],
    "efficiency_score": <float 0-1, where 1=very efficient>
}}

Focus on concrete evidence from the session data. Be specific about how the agent wasted time.
"""
        
        try:
            # Use the existing LM infrastructure
            response = await self.lm.respond_async(
                system_message="You are an expert evaluator analyzing AI agent efficiency in games. Respond only with valid JSON.",
                user_message=prompt
            )
            
            print(f"DEBUG - Raw LLM response: {response.raw_response[:200]}...")
            
            # Parse JSON response
            try:
                result = json.loads(response.raw_response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.raw_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "score": 0.5,
                        "reasoning": "Could not parse LLM response",
                        "evidence": ["Response parsing failed"],
                        "inefficiencies": [],
                        "efficiency_score": 0.5
                    }
            
            return [Judgement(
                criteria="wasted_time",
                score=result.get("score", 0.5),
                reasoning=result.get("reasoning", "No reasoning provided"),
                evidence=result.get("evidence", [])
            )]
            
        except Exception as e:
            return [Judgement(
                criteria="wasted_time",
                score=0.5,
                reasoning=f"Evaluation failed: {str(e)}",
                evidence=[f"Error: {str(e)}"]
            )]

class CrafterTraceAnalyzer:
    """Main analyzer for Crafter traces."""
    
    def __init__(self, experiment_id: str, db_path: str = "crafter_traces.duckdb", model_name: str = "gpt-4o-mini"):
        self.experiment_id = experiment_id
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize evaluators
        self.misunderstood_rules_eval = MisunderstoodCrafterRulesEval(model_name)
        self.wasted_time_eval = WastedTimeEval(model_name)
    
    def extract_session_data(self, session_id: str) -> Dict[str, Any]:
        """Extract session data from DuckDB."""
        conn = duckdb.connect(self.db_path)
        
        # Get session metadata
        result = conn.execute("""
            SELECT metadata FROM session_traces 
            WHERE session_id = ? AND experiment_id = ?
        """, [session_id, self.experiment_id]).fetchall()
        
        session_data = {}
        if result:
            metadata = json.loads(result[0][0]) if isinstance(result[0][0], str) else result[0][0]
            
            # Extract achievements
            for item in metadata:
                if item.get('metadata_type') == 'SessionMetadum':
                    data = item.get('data', {})
                    if 'achievements' in data:
                        achievements = data['achievements']
                        session_data['achievements'] = [k for k, v in achievements.items() if v]
                    if 'num_achievements' in data:
                        session_data['num_achievements'] = data['num_achievements']
                    if 'total_reward' in data:
                        session_data['total_reward'] = data['total_reward']
                    if 'rollout_length' in data:
                        session_data['total_steps'] = data['rollout_length']
        
        # Get events for action analysis
        result = conn.execute("""
            SELECT event_type, metadata, event_metadata
            FROM events 
            WHERE session_id = ? 
            ORDER BY event_time
        """, [session_id]).fetchall()
        
        actions = []
        invalid_actions = []
        inventory_changes = []
        
        for event_type, metadata, event_metadata in result:
            if event_type == 'runtime':
                # Parse action from metadata
                try:
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    if 'action' in meta_dict:
                        actions.append(meta_dict['action'])
                except:
                    pass
                
                # Check for invalid actions in event_metadata
                if event_metadata:
                    try:
                        hook_list = json.loads(event_metadata) if isinstance(event_metadata, str) else event_metadata
                        for hook_str in hook_list:
                            if isinstance(hook_str, str):
                                import re
                                if "'hook_name': 'invalid_action'" in hook_str:
                                    # Extract the actual action name from the description
                                    action_match = re.search(r"'action':\s*'([^']+)'", hook_str)
                                    if action_match:
                                        invalid_actions.append(action_match.group(1))
                                    else:
                                        invalid_actions.append("unknown")
                    except:
                        pass
            
            elif event_type == 'environment':
                # Check for inventory changes
                if event_metadata:
                    try:
                        hook_list = json.loads(event_metadata) if isinstance(event_metadata, str) else event_metadata
                        for hook_str in hook_list:
                            if isinstance(hook_str, str):
                                if "'hook_name': 'inventory_increase'" in hook_str:
                                    inventory_changes.append("inventory_increase")
                    except:
                        pass
        
        session_data.update({
            'actions': actions,
            'invalid_actions': invalid_actions,
            'inventory_changes': inventory_changes
        })
        
        conn.close()
        return session_data
    
    async def evaluate_session(self, session_id: str) -> Dict[str, List[Judgement]]:
        """Evaluate a single session."""
        session_data = self.extract_session_data(session_id)
        
        # Run evaluations
        misunderstood_rules = await self.misunderstood_rules_eval.run(session_data)
        wasted_time = await self.wasted_time_eval.run(session_data)
        
        return {
            'misunderstood_rules': misunderstood_rules,
            'wasted_time': wasted_time
        }
    
    async def evaluate_experiment(self) -> Dict[str, Any]:
        """Evaluate all sessions in the experiment."""
        conn = duckdb.connect(self.db_path)
        
        # Get all session IDs for this experiment
        result = conn.execute("""
            SELECT session_id FROM session_traces 
            WHERE experiment_id = ?
            ORDER BY session_id
        """, [self.experiment_id]).fetchall()
        
        session_ids = [row[0] for row in result]
        conn.close()
        
        print(f"Evaluating {len(session_ids)} sessions in parallel...")
        
        # Create all evaluation tasks
        tasks = [self.evaluate_session(session_id) for session_id in session_ids]
        
        # Run all evaluations in parallel
        all_results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        all_results = {}
        for session_id, results in zip(session_ids, all_results_list):
            all_results[session_id] = results
        
        return all_results

async def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
    else:
        print("Usage: python custom_eval_pipelines.py <experiment_id>")
        print("Example: python custom_eval_pipelines.py 77022cce-4bda-4415-9bce-0095e4ef2237")
        return
    
    # Use Gemini for evaluation
    analyzer = CrafterTraceAnalyzer(experiment_id, model_name="gemini-1.5-flash")
    results = await analyzer.evaluate_experiment()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for session_id, session_results in results.items():
        print(f"\nSession: {session_id}")
        print("-" * 60)
        
        for eval_type, judgements in session_results.items():
            print(f"\n{eval_type.upper()}:")
            for judgement in judgements:
                print(f"  Score: {judgement.score:.3f}")
                print(f"  Reasoning: {judgement.reasoning}")
                print(f"  Evidence: {judgement.evidence}")

if __name__ == "__main__":
    asyncio.run(main())
