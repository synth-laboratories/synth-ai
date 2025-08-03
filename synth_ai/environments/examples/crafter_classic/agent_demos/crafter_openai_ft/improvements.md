Ran tool
Ran tool
Ran tool
Read file: synth-ai/synth_ai/environments/examples/crafter_classic/agent_demos/crafter_openai_ft/filter_traces_sft_duckdb.py
Ran tool
Read file: synth-ai/synth_ai/environments/examples/crafter_classic/agent_demos/crafter_openai_ft/compare_experiments.py
Here are the **core synth_ai files** used by this code:

## **Core Synth-AI Files Used:**

### **Tracing V2 System:**
- `synth_ai/tracing_v2/duckdb/manager.py` - DuckDB trace manager
- `synth_ai/tracing_v2/duckdb/ft_utils.py` - Fine-tuning data extractor
- `synth_ai/tracing_v2/session_tracer.py` - Session tracing
- `synth_ai/tracing_v2/abstractions.py` - CAISEvent abstractions
- `synth_ai/tracing_v2/utils.py` - Experiment context utilities

### **LM Core System:**
- `synth_ai/lm/core/main_v2.py` - LM class with v2 tracing
- `synth_ai/lm/core/vendor_clients.py` - Vendor client management

### **Environments System:**
- `synth_ai/environments/examples/crafter_classic/trace_hooks.py` - Crafter trace hooks
- `synth_ai/environments/examples/crafter_classic/taskset.py` - Crafter task definitions
- `synth_ai/environments/tasks/core.py` - Task core abstractions (Impetus, Intent)

### **Evaluation System:**
- `synth_ai/evals/base.py` - Base evaluation classes (Judgement, BaseEval)

### **UI System:**
- `synth_ai/tui/cli/query_experiments.py` - Experiment query CLI

### **Provider Support:**
- `synth_ai/lm/provider_support/openai.py` - OpenAI provider (via logging references)

That's it. The main dependencies are the **tracing_v2 system** for database management and the **environments system** for Crafter-specific functionality.

# Improvements and Rough Edges Analysis

## 🚨 Critical Issues Found

### 1. **Hardcoded Experiment IDs in compare_experiments.py**
**Problem:** The script had hardcoded experiment IDs that didn't match our actual runs.
```python
# OLD - Hardcoded wrong IDs
EXPERIMENTS = {
    "gpt-4.1-nano": "194a3cd2-ecd3-4081-b46d-a7883e4a86f9",
    "gpt-4.1-mini": "da74a769-b33d-4b60-ae2a-52a4b67b3f35"
}

# FIXED - Updated to actual experiment IDs
EXPERIMENTS = {
    "gpt-4o-mini": "137683ed-3bd5-4bd3-9162-dae0371ddd3d",
    "gpt-4o": "207307d5-4105-4a18-bb93-89936047fa18"
}
```

**Impact:** Script failed with KeyError until manually updated.

### 2. **Inconsistent Model Name References**
**Problem:** Script had multiple references to old model names that needed updating.
```python
# OLD - Multiple hardcoded references
print("🔍 COMPARING GPT-4.1-NANO vs GPT-4.1-MINI EXPERIMENTS")
return conn.execute(query, [EXPERIMENTS["gpt-4.1-nano"], EXPERIMENTS["gpt-4.1-mini"]]).df()

# FIXED - Updated all references
print("🔍 COMPARING GPT-4O-MINI vs GPT-4O EXPERIMENTS")
return conn.execute(query, [EXPERIMENTS["gpt-4o-mini"], EXPERIMENTS["gpt-4o"]]).df()
```

## 🔗 **Related Shared Abstractions**

Based on analysis of the codebase, here are the key shared abstractions that could be extracted and improved:

### 1. **Database Connection & Management**
**Current State:** Each script has its own database connection logic.
```python
# In compare_experiments.py
def connect_to_db():
    return duckdb.connect("crafter_traces.duckdb")

# In filter_traces_sft_duckdb.py  
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
```

**Proposed Shared Abstraction:**
```python
class ExperimentDatabaseManager:
    """Unified database management for experiment analysis."""
    
    def __init__(self, db_path: str = "crafter_traces.duckdb"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Establish database connection with error handling."""
        try:
            self.conn = duckdb.connect(self.db_path)
            return self.conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.db_path}: {e}")
    
    def get_experiments(self, limit: int = 10) -> pd.DataFrame:
        """Get recent experiments."""
        query = """
        SELECT id, name, created_at, description 
        FROM experiments 
        ORDER BY created_at DESC 
        LIMIT ?
        """
        return self.conn.execute(query, [limit]).df()
    
    def get_experiment_traces(self, experiment_id: str) -> pd.DataFrame:
        """Get all traces for an experiment."""
        query = """
        SELECT st.*, e.name as experiment_name
        FROM session_traces st
        JOIN experiments e ON st.experiment_id = e.id
        WHERE st.experiment_id = ?
        """
        return self.conn.execute(query, [experiment_id]).df()
```

### 2. **Achievement Extraction & Analysis**
**Current State:** Achievement parsing logic is duplicated across scripts.
```python
# In compare_experiments.py (lines 67-100)
def get_achievement_analysis(conn, experiment_id: str) -> Dict[str, Any]:
    # Complex achievement extraction logic
    for session_id, metadata in results:
        if metadata:
            metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
            for meta_item in metadata_list:
                if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                    data = meta_item.get('data', {})
                    if 'achievements' in data:
                        achievements_dict = data['achievements']
                        # ... more parsing logic

# In filter_traces_sft_duckdb.py (similar logic)
def extract_achievements(trace):
    achievements = trace.get('final_achievements', {})
    if isinstance(achievements, dict):
        return len([k for k, v in achievements.items() if v])
```

**Proposed Shared Abstraction:**
```python
class AchievementAnalyzer:
    """Standardized achievement extraction and analysis."""
    
    @staticmethod
    def extract_achievements_from_metadata(metadata: Union[str, Dict, List]) -> Dict[str, Any]:
        """Extract achievements from various metadata formats."""
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        achievements = {}
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict) and item.get('metadata_type') == 'SessionMetadum':
                    data = item.get('data', {})
                    if 'achievements' in data:
                        achievements.update(data['achievements'])
        elif isinstance(metadata, dict):
            achievements = metadata.get('final_achievements', {})
        
        return achievements
    
    @staticmethod
    def count_achievements(achievements: Dict[str, Any]) -> int:
        """Count unlocked achievements."""
        if isinstance(achievements, dict):
            return sum(1 for achieved in achievements.values() if achieved)
        elif isinstance(achievements, list):
            return len(achievements)
        return 0
    
    @staticmethod
    def get_achievement_types(achievements: Dict[str, Any]) -> List[str]:
        """Get list of achievement types."""
        if isinstance(achievements, dict):
            return list(achievements.keys())
        return []
```

### 3. **Trace Data Validation & Normalization**
**Current State:** Each script has its own data validation logic.
```python
# In filter_traces_sft_duckdb.py
def load_traces(trace_file: str) -> List[Dict[str, Any]]:
    with open(trace_file, 'r') as f:
        if trace_file.endswith('.json'):
            return json.load(f)
        elif trace_file.endswith('.jsonl'):
            return [json.loads(line) for line in f]

# In compare_experiments.py
def get_session_stats(conn, experiment_id: str) -> pd.DataFrame:
    # Custom query logic for session statistics
```

**Proposed Shared Abstraction:**
```python
class TraceDataValidator:
    """Validate and normalize trace data formats."""
    
    @staticmethod
    def validate_trace_structure(trace: Dict[str, Any]) -> bool:
        """Validate required fields in trace data."""
        required_fields = ['messages', 'final_achievements', 'shaped_reward']
        return all(field in trace for field in required_fields)
    
    @staticmethod
    def normalize_trace_format(trace: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trace to standard format."""
        # Ensure consistent field names and types
        normalized = {
            'messages': trace.get('messages', []),
            'final_achievements': trace.get('final_achievements', {}),
            'shaped_reward': float(trace.get('shaped_reward', 0.0)),
            'turns': trace.get('turns', []),
            'metadata': trace.get('metadata', {})
        }
        return normalized
    
    @staticmethod
    def load_traces_from_source(source_path: str) -> List[Dict[str, Any]]:
        """Load traces from various sources with validation."""
        if source_path.endswith('.json'):
            with open(source_path, 'r') as f:
                traces = json.load(f)
        elif source_path.endswith('.jsonl'):
            with open(source_path, 'r') as f:
                traces = [json.loads(line) for line in f]
        elif source_path.endswith('.duckdb'):
            # Use DuckDB manager
            manager = DuckDBTraceManager(source_path)
            traces = manager.query_traces("SELECT * FROM session_traces").to_dict('records')
        else:
            raise ValueError(f"Unsupported source format: {source_path}")
        
        # Validate and normalize
        validated_traces = []
        for trace in traces:
            if TraceDataValidator.validate_trace_structure(trace):
                validated_traces.append(TraceDataValidator.normalize_trace_format(trace))
        
        return validated_traces
```

### 4. **Metrics Calculation Framework**
**Current State:** Metrics calculation is scattered and inconsistent.
```python
# In compare_experiments.py
def analyze_instance_difficulty(conn) -> Dict[str, Any]:
    # Custom metrics calculation
    analysis[experiment_id] = {
        "total_sessions": len(exp_data),
        "avg_timesteps": exp_data['num_timesteps'].mean(),
        "avg_events": exp_data['num_events'].mean(),
        # ... more metrics
    }

# In filter_traces_sft_duckdb.py
def print_statistics(stats: Dict[str, Any]):
    # Custom statistics display
```

**Proposed Shared Abstraction:**
```python
class ExperimentMetricsCalculator:
    """Standardized metrics calculation for experiments."""
    
    @staticmethod
    def calculate_achievement_metrics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate standardized achievement metrics."""
        total_achievements = 0
        unique_achievements = set()
        achievement_counts = defaultdict(int)
        
        for trace in traces:
            achievements = AchievementAnalyzer.extract_achievements_from_metadata(
                trace.get('final_achievements', {})
            )
            count = AchievementAnalyzer.count_achievements(achievements)
            total_achievements += count
            
            # Track unique achievements
            for achievement, unlocked in achievements.items():
                if unlocked:
                    unique_achievements.add(achievement)
                    achievement_counts[achievement] += 1
        
        return {
            'total_achievements': total_achievements,
            'unique_achievements': len(unique_achievements),
            'avg_achievements_per_episode': total_achievements / len(traces) if traces else 0,
            'achievement_rate': sum(1 for t in traces if AchievementAnalyzer.count_achievements(
                AchievementAnalyzer.extract_achievements_from_metadata(t.get('final_achievements', {}))
            ) > 0) / len(traces) if traces else 0,
            'achievement_counts': dict(achievement_counts)
        }
    
    @staticmethod
    def calculate_performance_metrics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate standardized performance metrics."""
        if not traces:
            return {}
        
        rewards = [t.get('shaped_reward', 0.0) for t in traces]
        turns = [len(t.get('messages', [])) for t in traces]
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_turns': np.mean(turns),
            'std_turns': np.std(turns),
            'completion_rate': sum(1 for t in traces if t.get('final_achievements')) / len(traces),
            'total_episodes': len(traces)
        }
```

### 5. **Configuration Management**
**Current State:** Hardcoded values throughout scripts.
```python
# In compare_experiments.py
EXPERIMENTS = {
    "gpt-4o-mini": "137683ed-3bd5-4bd3-9162-dae0371ddd3d",
    "gpt-4o": "207307d5-4105-4a18-bb93-89936047fa18"
}

# In filter_traces_sft_duckdb.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    # ... many hardcoded defaults
```

**Proposed Shared Abstraction:**
```python
class ExperimentConfig:
    """Configuration management for experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return toml.load(f)
        
        return {
            'database': {
                'path': 'crafter_traces.duckdb',
                'backup_enabled': True
            },
            'filtering': {
                'min_achievements': 3,
                'min_reward': 10.0,
                'max_turns': 30,
                'achievement_types': []
            },
            'comparison': {
                'metrics': ['achievement_rate', 'avg_achievements_per_episode'],
                'visualization': True
            }
        }
    
    def get_experiment_ids(self) -> Dict[str, str]:
        """Get experiment IDs from config or database."""
        if 'experiments' in self.config:
            return self.config['experiments']
        
        # Fallback to database discovery
        db_manager = ExperimentDatabaseManager(self.config['database']['path'])
        experiments = db_manager.get_experiments()
        return {exp['name']: exp['id'] for _, exp in experiments.iterrows()}
```

### 6. **Visualization & Reporting**
**Current State:** Basic print statements and custom visualization.
```python
# In filter_traces_sft_duckdb.py
def create_histogram(data: List[float], bins: int = 20, width: int = 60, height: int = 15, 
                    title: str = "", x_label: str = "", y_label: str = "") -> str:
    # Custom ASCII histogram implementation

def print_statistics(stats: Dict[str, Any]):
    # Custom statistics display
```

**Proposed Shared Abstraction:**
```python
class ExperimentReporter:
    """Standardized reporting and visualization."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_comparison_report(self, exp1_metrics: Dict, exp2_metrics: Dict, 
                                 exp1_name: str, exp2_name: str) -> str:
        """Generate standardized comparison report."""
        report = f"""
# Experiment Comparison Report

## Overview
- **Experiment 1**: {exp1_name}
- **Experiment 2**: {exp2_name}
- **Generated**: {datetime.now().isoformat()}

## Achievement Metrics
| Metric | {exp1_name} | {exp2_name} | Improvement |
|--------|-------------|-------------|-------------|
| Total Achievements | {exp1_metrics['total_achievements']} | {exp2_metrics['total_achievements']} | {self._calculate_improvement(exp1_metrics['total_achievements'], exp2_metrics['total_achievements'])} |
| Achievement Rate | {exp1_metrics['achievement_rate']:.1%} | {exp2_metrics['achievement_rate']:.1%} | {self._calculate_improvement(exp1_metrics['achievement_rate'], exp2_metrics['achievement_rate'])} |
| Avg Achievements/Episode | {exp1_metrics['avg_achievements_per_episode']:.2f} | {exp2_metrics['avg_achievements_per_episode']:.2f} | {self._calculate_improvement(exp1_metrics['avg_achievements_per_episode'], exp2_metrics['avg_achievements_per_episode'])} |

## Performance Metrics
| Metric | {exp1_name} | {exp2_name} | Improvement |
|--------|-------------|-------------|-------------|
| Avg Reward | {exp1_metrics['avg_reward']:.2f} | {exp2_metrics['avg_reward']:.2f} | {self._calculate_improvement(exp1_metrics['avg_reward'], exp2_metrics['avg_reward'])} |
| Completion Rate | {exp1_metrics['completion_rate']:.1%} | {exp2_metrics['completion_rate']:.1%} | {self._calculate_improvement(exp1_metrics['completion_rate'], exp2_metrics['completion_rate'])} |
"""
        return report
    
    def _calculate_improvement(self, old_val: float, new_val: float) -> str:
        """Calculate percentage improvement."""
        if old_val == 0:
            return "N/A" if new_val == 0 else "∞"
        improvement = ((new_val - old_val) / old_val) * 100
        return f"{improvement:+.1f}%"
```

## 🎯 **Implementation Priority**

### **High Priority (Immediate Impact)**
1. **ExperimentDatabaseManager** - Eliminates hardcoded experiment IDs
2. **AchievementAnalyzer** - Standardizes achievement extraction
3. **TraceDataValidator** - Ensures data consistency

### **Medium Priority (Quality of Life)**
4. **ExperimentMetricsCalculator** - Standardizes metrics
5. **ExperimentConfig** - Eliminates hardcoded values
6. **ExperimentReporter** - Improves output quality

### **Low Priority (Nice to Have)**
7. **Advanced visualization** - Interactive plots
8. **Web dashboard** - Real-time monitoring
9. **Automated testing** - Unit tests for all abstractions

## 🔗 **Logic Dependencies & Abstractions Analysis**

### **What Logic These Files Draw On:**

#### **1. Synth-AI Tracing V2 System**
**Primary Dependencies:**
```python
# filter_traces_sft_duckdb.py
from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager

# compare_experiments.py (indirect dependency)
# Uses DuckDB schema defined by tracing_v2 system
```

**Key Abstractions Used:**
- **`DuckDBTraceManager`** - Database management for trace storage
- **`FinetuningDataExtractor`** - Utilities for extracting fine-tuning data
- **DuckDB Schema** - Standardized database schema for traces

#### **2. Synth-AI LM Core System**
**Indirect Dependencies:**
```python
# test_crafter_react_agent_openai.py
from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import SessionTracer
from synth_ai.tracing_v2.abstractions import CAISEvent
```

**Key Abstractions Used:**
- **`LM` class** - Language model interface with v2 tracing
- **`SessionTracer`** - Session-based tracing orchestration
- **`CAISEvent`** - AI/LLM event abstractions

#### **3. Synth-AI Environments System**
**Indirect Dependencies:**
```python
# test_crafter_react_agent_openai.py
from synth_ai.environments.examples.crafter_classic.trace_hooks import CRAFTER_HOOKS
from synth_ai.environments.examples.crafter_classic.taskset import (
    CRAFTER_TASKS, CRAFTER_TASK_NAMES
)
```

**Key Abstractions Used:**
- **Environment Hooks** - Trace hooks for Crafter environment
- **Task Definitions** - Crafter-specific task abstractions

### **Abstractions That Could Be Improved:**

#### **1. Database Schema Abstraction**
**Current Issue:** Hardcoded SQL queries scattered across scripts.
```python
# compare_experiments.py - Lines 25-35
query = """
SELECT 
    e.id, e.name, e.description, e.created_at,
    sv.branch, sv.commit
FROM experiments e
LEFT JOIN experimental_systems es ON e.id = es.experiment_id
LEFT JOIN system_versions sv ON es.system_version_id = sv.id
WHERE e.id = ?
"""
```

**Proposed Improvement:**
```python
class ExperimentQueryBuilder:
    """Type-safe query builder for experiment analysis."""
    
    @staticmethod
    def get_experiment_summary(experiment_id: str) -> str:
        """Get standardized experiment summary query."""
        return """
        SELECT 
            e.id, e.name, e.description, e.created_at,
            sv.branch, sv.commit
        FROM experiments e
        LEFT JOIN experimental_systems es ON e.id = es.experiment_id
        LEFT JOIN system_versions sv ON es.system_version_id = sv.id
        WHERE e.id = ?
        """
    
    @staticmethod
    def get_session_stats(experiment_id: str) -> str:
        """Get standardized session statistics query."""
        return """
        SELECT 
            st.session_id, st.created_at, st.num_timesteps,
            st.num_events, st.num_messages, st.metadata
        FROM session_traces st
        WHERE st.experiment_id = ?
        ORDER BY st.created_at
        """
```

#### **2. Metadata Parsing Abstraction**
**Current Issue:** Complex metadata parsing logic duplicated.
```python
# compare_experiments.py - Lines 67-100
for session_id, metadata in results:
    if metadata:
        metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
        for meta_item in metadata_list:
            if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                data = meta_item.get('data', {})
                if 'achievements' in data:
                    achievements_dict = data['achievements']
                    # ... complex parsing logic
```

**Proposed Improvement:**
```python
class MetadataParser:
    """Standardized metadata parsing for trace data."""
    
    @staticmethod
    def parse_session_metadata(metadata: Union[str, Dict, List]) -> Dict[str, Any]:
        """Parse session metadata with error handling."""
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        
        if isinstance(metadata, list):
            return MetadataParser._parse_metadata_list(metadata)
        elif isinstance(metadata, dict):
            return MetadataParser._parse_metadata_dict(metadata)
        
        return {}
    
    @staticmethod
    def _parse_metadata_list(metadata_list: List[Dict]) -> Dict[str, Any]:
        """Parse list of metadata items."""
        parsed = {}
        for item in metadata_list:
            if isinstance(item, dict) and item.get('metadata_type') == 'SessionMetadum':
                data = item.get('data', {})
                parsed.update(data)
        return parsed
    
    @staticmethod
    def _parse_metadata_dict(metadata_dict: Dict) -> Dict[str, Any]:
        """Parse single metadata dictionary."""
        return metadata_dict.get('data', {})
```

#### **3. Experiment Discovery Abstraction**
**Current Issue:** No standardized way to discover experiments.
```python
# compare_experiments.py - Hardcoded experiment IDs
EXPERIMENTS = {
    "gpt-4o-mini": "137683ed-3bd5-4bd3-9162-dae0371ddd3d",
    "gpt-4o": "207307d5-4105-4a18-bb93-89936047fa18"
}
```

**Proposed Improvement:**
```python
class ExperimentDiscovery:
    """Dynamic experiment discovery and management."""
    
    def __init__(self, db_manager: DuckDBTraceManager):
        self.db_manager = db_manager
    
    def get_recent_experiments(self, limit: int = 10) -> pd.DataFrame:
        """Get recent experiments from database."""
        query = """
        SELECT id, name, created_at, description, model_name
        FROM experiments 
        ORDER BY created_at DESC 
        LIMIT ?
        """
        return self.db_manager.query_traces(query, [limit])
    
    def find_experiments_by_model(self, model_name: str) -> pd.DataFrame:
        """Find experiments by model name."""
        query = """
        SELECT id, name, created_at, description
        FROM experiments 
        WHERE name LIKE ?
        ORDER BY created_at DESC
        """
        return self.db_manager.query_traces(query, [f"%{model_name}%"])
    
    def get_experiment_pairs(self, model1: str, model2: str) -> Dict[str, str]:
        """Get experiment pairs for comparison."""
        experiments = self.get_recent_experiments(limit=20)
        model1_exps = experiments[experiments['name'].str.contains(model1, na=False)]
        model2_exps = experiments[experiments['name'].str.contains(model2, na=False)]
        
        if len(model1_exps) > 0 and len(model2_exps) > 0:
            return {
                model1: model1_exps.iloc[0]['id'],
                model2: model2_exps.iloc[0]['id']
            }
        
        raise ValueError(f"Could not find experiments for {model1} and {model2}")
```

#### **4. Metrics Calculation Abstraction**
**Current Issue:** Metrics calculation logic is scattered and inconsistent.
```python
# compare_experiments.py - Lines 147-198
def get_session_performance_comparison(conn) -> pd.DataFrame:
    # Custom performance comparison logic
    query = """
    SELECT 
        st.experiment_id,
        COUNT(*) as total_sessions,
        AVG(st.num_timesteps) as avg_timesteps,
        AVG(st.num_events) as avg_events
    FROM session_traces st
    WHERE st.experiment_id IN (?, ?)
    GROUP BY st.experiment_id
    """
```

**Proposed Improvement:**
```python
class MetricsCalculator:
    """Standardized metrics calculation for experiments."""
    
    @staticmethod
    def calculate_session_metrics(conn, experiment_ids: List[str]) -> pd.DataFrame:
        """Calculate standardized session metrics."""
        placeholders = ','.join(['?' for _ in experiment_ids])
        query = f"""
        SELECT 
            st.experiment_id,
            COUNT(*) as total_sessions,
            AVG(st.num_timesteps) as avg_timesteps,
            AVG(st.num_events) as avg_events,
            AVG(st.num_messages) as avg_messages,
            SUM(CASE WHEN st.metadata IS NOT NULL THEN 1 ELSE 0 END) as sessions_with_metadata
        FROM session_traces st
        WHERE st.experiment_id IN ({placeholders})
        GROUP BY st.experiment_id
        """
        return conn.execute(query, experiment_ids).df()
    
    @staticmethod
    def calculate_achievement_metrics(conn, experiment_ids: List[str]) -> pd.DataFrame:
        """Calculate standardized achievement metrics."""
        # Implementation with proper metadata parsing
        pass
    
    @staticmethod
    def calculate_model_usage_metrics(conn, experiment_ids: List[str]) -> pd.DataFrame:
        """Calculate standardized model usage metrics."""
        # Implementation with proper event analysis
        pass
```

#### **5. Visualization Abstraction**
**Current Issue:** Custom visualization logic in each script.
```python
# filter_traces_sft_duckdb.py - Lines 25-100
def create_histogram(data: List[float], bins: int = 20, width: int = 60, height: int = 15, 
                    title: str = "", x_label: str = "", y_label: str = "") -> str:
    # Custom ASCII histogram implementation
```

**Proposed Improvement:**
```python
class ExperimentVisualizer:
    """Standardized visualization for experiment analysis."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_histogram(self, data: List[float], **kwargs) -> str:
        """Create standardized histogram."""
        # Reuse existing implementation with better defaults
        return create_histogram(data, **kwargs)
    
    def create_comparison_chart(self, exp1_data: Dict, exp2_data: Dict, 
                               exp1_name: str, exp2_name: str) -> str:
        """Create standardized comparison chart."""
        # Implementation for comparing two experiments
        pass
    
    def save_visualization(self, content: str, filename: str):
        """Save visualization to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Visualization saved to {output_path}")
```

### **Integration with Existing Synth-AI Abstractions:**

#### **1. Leverage Existing `DuckDBTraceManager`**
**Current Usage:**
```python
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor
```

**Proposed Enhancement:**
```python
class EnhancedDuckDBManager(DuckDBTraceManager):
    """Enhanced DuckDB manager with experiment analysis capabilities."""
    
    def get_experiment_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment analysis."""
        # Use existing query_traces method
        sessions = self.query_traces(
            "SELECT * FROM session_traces WHERE experiment_id = ?",
            [experiment_id]
        )
        
        # Add analysis methods
        return {
            'session_count': len(sessions),
            'avg_timesteps': sessions['num_timesteps'].mean(),
            'avg_events': sessions['num_events'].mean(),
            'metadata_analysis': self._analyze_metadata(sessions)
        }
    
    def _analyze_metadata(self, sessions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze session metadata."""
        # Implementation using existing metadata parsing
        pass
```

#### **2. Extend `FinetuningDataExtractor`**
**Current Usage:**
```python
extractor = FinetuningDataExtractor("traces.duckdb")
successful_sessions = extractor.get_successful_sessions(min_reward=10.0)
```

**Proposed Enhancement:**
```python
class EnhancedFinetuningExtractor(FinetuningDataExtractor):
    """Enhanced fine-tuning data extractor with experiment analysis."""
    
    def get_experiment_comparison_data(self, exp1_id: str, exp2_id: str) -> Dict[str, Any]:
        """Get comparison data for two experiments."""
        exp1_sessions = self.get_successful_sessions_for_experiment(exp1_id)
        exp2_sessions = self.get_successful_sessions_for_experiment(exp2_id)
        
        return {
            'experiment_1': {
                'id': exp1_id,
                'sessions': exp1_sessions,
                'metrics': self._calculate_metrics(exp1_sessions)
            },
            'experiment_2': {
                'id': exp2_id,
                'sessions': exp2_sessions,
                'metrics': self._calculate_metrics(exp2_sessions)
            }
        }
    
    def _calculate_metrics(self, sessions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate standardized metrics for sessions."""
        # Implementation using existing methods
        pass
```

### **Key Benefits of These Improvements:**

1. **Eliminates Hardcoded Values** - Dynamic experiment discovery
2. **Standardizes Data Access** - Consistent database queries
3. **Improves Error Handling** - Robust metadata parsing
4. **Enables Reusability** - Shared abstractions across scripts
5. **Enhances Maintainability** - Centralized logic in reusable classes
6. **Facilitates Testing** - Mockable abstractions for unit tests
7. **Supports OSS Release** - Clean, documented abstractions

This analysis shows that while the core functionality works, there are significant opportunities to make the codebase more robust, maintainable, and user-friendly through shared abstractions.
