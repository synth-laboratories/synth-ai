#!/usr/bin/env python3
"""
================================================================================
GRAPHGEN DEMO: Banking77 Intent Classification
================================================================================

This demo shows how to use Synth's GraphGen to automatically generate and
optimize an LLM workflow (graph) for a classification task.

WHAT IS GRAPHGEN?
-----------------
GraphGen is a system that:
1. Takes a dataset of (input, expected_output) pairs
2. Automatically proposes LLM workflow architectures ("graphs")
3. Evaluates each graph on your data using a verifier
4. Evolves and improves the graphs over multiple generations
5. Returns the best-performing graph ready for production inference

KEY CONCEPTS:
- Task: A single (input, expected_output) pair for training/evaluation
- Graph: An LLM workflow that processes inputs to produce outputs
- Verifier: Evaluates if a graph's output matches the expected output
- Generation: One round of proposing and evaluating graph candidates

USAGE:
    uv run python demos/graphgen_banking77/run_demo.py
"""

# ==============================================================================
# STEP 0: IMPORTS AND CLI CONFIGURATION
# ==============================================================================

import random
import time

import httpx
from datasets import load_dataset
from synth_ai.core.urls import BACKEND_URL_BASE, backend_health_url
from synth_ai.sdk.api.train.graphgen import GraphGenJob
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenGoldOutput,
    GraphGenTask,
    GraphGenTaskSet,
    GraphGenTaskSetMetadata,
    GraphGenVerifierConfig,
)
from synth_ai.sdk.auth import get_or_mint_synth_api_key

# ==============================================================================
# STEP 1: BACKEND CONFIGURATION
# ==============================================================================
# GraphGen runs against the Synth backend. Use SYNTH_BACKEND_URL to override.


def setup_backend() -> str:
    """Configure and verify the backend connection.

    Returns:
        The backend base URL
    """
    backend_url = BACKEND_URL_BASE
    print("=" * 60)
    print("BACKEND")
    print(f"Backend: {backend_url}")
    print("=" * 60)

    # Verify backend is healthy before proceeding
    print("\nChecking backend health...")
    try:
        r = httpx.get(backend_health_url(backend_url), timeout=30)
        if r.status_code == 200:
            print(f"  Backend healthy: {r.json()}")
        else:
            raise RuntimeError(f"Backend returned status {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Could not connect to backend: {e}")
        raise

    return backend_url


# ==============================================================================
# STEP 2: API KEY CONFIGURATION
# ==============================================================================
# You need an API key to authenticate with the Synth backend.
# - Set SYNTH_API_KEY environment variable for production use
# - Or we'll mint a temporary demo key for testing


def setup_api_key() -> str:
    """Get or create an API key for authentication.

    Args:
    Returns:
        The API key string
    """
    api_key = get_or_mint_synth_api_key(backend_url=BACKEND_URL_BASE)
    print(f"\nUsing SYNTH_API_KEY: {api_key[:20]}...")
    return api_key


# ==============================================================================
# STEP 3: BUILD THE DATASET
# ==============================================================================
# GraphGen requires a structured dataset with:
#
# REQUIRED COMPONENTS:
# 1. Tasks: List of inputs to process (e.g., customer queries)
# 2. Gold Outputs: Expected outputs for each task (e.g., intent labels)
# 3. Input Schema: JSON schema defining the input format
# 4. Output Schema: JSON schema defining the output format
#
# OPTIONAL COMPONENTS:
# 5. Verifier Config: How to evaluate graph outputs (rubric, exact_match, etc.)
# 6. Problem Spec: Natural language description for the graph proposer


def build_banking77_graphgen_dataset(
    num_train_tasks: int = 50, num_test_tasks: int = 20
) -> tuple[GraphGenTaskSet, list[str]]:
    """Build a GraphGenTaskSet from the Banking77 dataset.

    Banking77 is a standard benchmark for intent classification with 77 classes.
    We'll use a subset of it to demonstrate GraphGen.

    Args:
        num_train_tasks: Number of training examples
        num_test_tasks: Number of test examples

    Returns:
        Tuple of (GraphGenTaskSet, label_names list)
    """
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING DATASET")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 3a. Load the raw dataset from HuggingFace
    # -------------------------------------------------------------------------
    print("\n[3a] Loading Banking77 from HuggingFace...")
    train_ds = load_dataset("banking77", split="train", trust_remote_code=False)
    test_ds = load_dataset("banking77", split="test", trust_remote_code=False)

    # Extract label names (the 77 intent categories)
    label_names = (
        train_ds.features["label"].names if hasattr(train_ds.features.get("label"), "names") else []
    )

    print(f"  Train size: {len(train_ds)}")
    print(f"  Test size: {len(test_ds)}")
    print(f"  Intent labels: {len(label_names)}")

    # -------------------------------------------------------------------------
    # 3b. Sample a diverse subset of examples
    # -------------------------------------------------------------------------
    # We shuffle indices from the full dataset to get diverse classes,
    # not just sequential examples which might be clustered by class.
    print("\n[3b] Sampling diverse subset...")

    total_dataset_size = min(10000, len(train_ds))
    total_test_size = min(5000, len(test_ds))

    # Shuffle all indices, then take the first N
    all_train_indices = list(range(total_dataset_size))
    all_test_indices = list(range(total_test_size))
    random.shuffle(all_train_indices)
    random.shuffle(all_test_indices)

    selected_train_indices = all_train_indices[:num_train_tasks]
    selected_test_indices = all_test_indices[:num_test_tasks]

    # -------------------------------------------------------------------------
    # 3c. Convert raw examples to GraphGen format
    # -------------------------------------------------------------------------
    print("\n[3c] Converting to GraphGen format...")

    all_examples = []

    # Process training examples
    for dataset_idx in selected_train_indices:
        row = train_ds[dataset_idx]
        label_idx = int(row.get("label", 0))
        label_text = (
            label_names[label_idx] if label_idx < len(label_names) else f"label_{label_idx}"
        )
        all_examples.append(
            {
                "query": str(row.get("text", "")),
                "intent": label_text,
                "source": "train",
            }
        )

    # Process test examples
    for dataset_idx in selected_test_indices:
        row = test_ds[dataset_idx]
        label_idx = int(row.get("label", 0))
        label_text = (
            label_names[label_idx] if label_idx < len(label_names) else f"label_{label_idx}"
        )
        all_examples.append(
            {
                "query": str(row.get("text", "")),
                "intent": label_text,
                "source": "test",
            }
        )

    # Shuffle to mix train/test
    random.shuffle(all_examples)

    # -------------------------------------------------------------------------
    # 3d. Create GraphGenTask and GraphGenGoldOutput objects
    # -------------------------------------------------------------------------
    # Each task has:
    #   - id: Unique identifier
    #   - input: The data to process (must match input_schema)
    #
    # Each gold output has:
    #   - task_id: Links to the corresponding task
    #   - output: The expected result (must match output_schema)

    print("\n[3d] Creating Task and GoldOutput objects...")

    tasks = []
    gold_outputs = []

    for task_idx, example in enumerate(all_examples):
        task_id = f"task_{task_idx}"

        # The task contains only the INPUT (what the graph will receive)
        tasks.append(
            GraphGenTask(
                id=task_id,
                input={"query": example["query"]},
            )
        )

        # The gold output contains the EXPECTED OUTPUT (for evaluation)
        gold_outputs.append(
            GraphGenGoldOutput(
                task_id=task_id,
                output={"intent": example["intent"]},
            )
        )

    # -------------------------------------------------------------------------
    # 3e. Define input and output schemas
    # -------------------------------------------------------------------------
    # JSON Schemas that define the structure of inputs and outputs.
    # These help GraphGen understand your data format and generate
    # appropriate graph architectures.

    print("\n[3e] Defining schemas...")

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Customer banking query to classify"}
        },
        "required": ["query"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Predicted banking intent label",
                "enum": label_names,  # Constrain to valid labels
            }
        },
        "required": ["intent"],
    }

    # -------------------------------------------------------------------------
    # 3f. Configure the verifier
    # -------------------------------------------------------------------------
    # The verifier evaluates how well a graph's output matches the gold output.
    # Options:
    #   - "rubric": LLM-based evaluation (flexible, good for subjective tasks)
    #   - "exact_match": String equality (strict, good for classification)
    #   - "code": Custom Python evaluation function

    print("\n[3f] Configuring verifier...")

    verifier_config = GraphGenVerifierConfig(
        mode="rubric",  # Use LLM to judge correctness
        model="gpt-4o-mini",  # Fast, cheap model for verification
        provider="openai",
    )

    # -------------------------------------------------------------------------
    # 3g. Assemble the complete GraphGenTaskSet
    # -------------------------------------------------------------------------
    print("\n[3g] Assembling GraphGenTaskSet...")

    dataset = GraphGenTaskSet(
        # Metadata describes the overall task
        metadata=GraphGenTaskSetMetadata(
            name="Banking77 Intent Classification",
            description="Banking77 dataset for intent classification graph optimization",
            input_schema=input_schema,
            output_schema=output_schema,
        ),
        # The actual data
        tasks=tasks,
        gold_outputs=gold_outputs,
        # How to evaluate outputs
        verifier_config=verifier_config,
        # Schema references (for backward compatibility)
        input_schema=input_schema,
        output_schema=output_schema,
        # Extract only "intent" field from graph output for evaluation
        # (graphs may produce extra fields; this tells the verifier what to compare)
        select_output="intent",
    )

    print("\n  Dataset ready!")
    print(f"  Total tasks: {len(tasks)} ({num_train_tasks} train, {num_test_tasks} test)")
    print(f"  Gold outputs: {len(gold_outputs)}")
    print(f"  Verifier mode: {verifier_config.mode}")

    return dataset, label_names


# ==============================================================================
# STEP 4: CREATE AND CONFIGURE THE GRAPHGEN JOB
# ==============================================================================
# A GraphGenJob orchestrates the entire optimization process.
#
# KEY CONFIGURATION OPTIONS:
# - policy_models: The LLM(s) used inside the generated graphs
# - rollout_budget: Max number of graph evaluations
# - proposer_effort: How hard the proposer tries ("low", "medium", "high")
# - num_generations: Number of evolution rounds
# - problem_spec: Natural language description of the task


def create_graphgen_job(
    dataset: GraphGenTaskSet,
    label_names: list[str],
    backend_url: str,
    api_key: str,
) -> GraphGenJob:
    """Create and configure a GraphGen optimization job.

    Args:
        dataset: The prepared GraphGenTaskSet
        label_names: List of valid intent labels (for problem spec)
        backend_url: Backend API URL
        api_key: Authentication key

    Returns:
        Configured GraphGenJob ready to submit
    """
    print("\n" + "=" * 60)
    print("STEP 4: CREATING GRAPHGEN JOB")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 4a. Write the problem specification
    # -------------------------------------------------------------------------
    # The problem_spec is a natural language description that helps the
    # graph proposer understand what kind of workflow to generate.
    # Be specific about:
    #   - What the task is
    #   - What the valid outputs are
    #   - Any constraints or requirements

    print("\n[4a] Writing problem specification...")

    problem_spec = (
        "You are building a banking intent classification system. "
        "Given a customer query, classify it into one of 77 banking intents. "
        "The system should use a single LLM call to analyze the query and return the intent label. "
        "\n\n"
        "VALID INTENT LABELS (must return exactly one of these):\n"
        + "\n".join([f"  - {label}" for label in label_names])
        + "\n\n"
        "The output must be a JSON object with an 'intent' field containing one of the above labels."
    )

    print(f"  Problem spec length: {len(problem_spec)} chars")

    # -------------------------------------------------------------------------
    # 4b. Configure and create the job
    # -------------------------------------------------------------------------
    print("\n[4b] Configuring job parameters...")

    job = GraphGenJob.from_dataset(
        dataset=dataset,
        # The LLM(s) to use inside generated graphs
        policy_models="gpt-4.1-mini",
        # Maximum graph evaluations (higher = more exploration, more cost)
        rollout_budget=100,
        # How creative/thorough the proposer is ("low", "medium", "high")
        proposer_effort="medium",
        # Number of evolution generations
        num_generations=2,
        # Natural language task description
        problem_spec=problem_spec,
        # Backend connection
        backend_url=backend_url,
        api_key=api_key,
        # Start immediately after creation
        auto_start=True,
    )

    print("\n  Job configured:")
    print(f"    Graph type: {job.config.graph_type}")
    print(f"    Policy models: {job.config.policy_models}")
    print(f"    Rollout budget: {job.config.rollout_budget}")
    print(f"    Proposer effort: {job.config.proposer_effort}")
    print(f"    Generations: {job.config.num_generations}")

    return job


# ==============================================================================
# STEP 5: SUBMIT AND MONITOR THE JOB
# ==============================================================================
# Once submitted, GraphGen will:
# 1. Generate initial graph candidates based on your problem_spec
# 2. Evaluate each candidate on your dataset
# 3. Select the best performers and evolve them
# 4. Repeat for num_generations
# 5. Return the best-performing graph


def submit_and_monitor(job: GraphGenJob) -> tuple[dict, float]:
    """Submit the job and stream progress until completion.

    Args:
        job: The configured GraphGenJob

    Returns:
        Tuple of (result dict, duration in seconds)
    """
    print("\n" + "=" * 60)
    print("STEP 5: RUNNING OPTIMIZATION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 5a. Submit the job
    # -------------------------------------------------------------------------
    print("\n[5a] Submitting job to backend...")
    job_id = job.submit()
    print(f"  Job ID: {job_id}")

    # -------------------------------------------------------------------------
    # 5b. Stream events until completion
    # -------------------------------------------------------------------------
    # The job runs asynchronously. We can poll for status or stream events.
    # stream_until_complete() handles both and prints progress.

    print("\n[5b] Streaming optimization events...")
    print("     (This may take several minutes)\n")

    start_time = time.time()

    result = job.stream_until_complete(
        timeout=3600.0,  # Max 1 hour
        interval=3.0,  # Poll every 3 seconds
    )

    duration = time.time() - start_time

    return result, duration


# ==============================================================================
# STEP 6: VIEW RESULTS AND USE THE OPTIMIZED GRAPH
# ==============================================================================
# After optimization, you can:
# - View the best score achieved
# - Download the optimized graph specification
# - Run inference with the optimized graph


def display_results(job: GraphGenJob, result: dict, duration: float):
    """Display optimization results and demonstrate inference.

    Args:
        job: The completed GraphGenJob
        result: Result dictionary from stream_until_complete
        duration: How long optimization took
    """
    print("\n" + "=" * 60)
    print("STEP 6: RESULTS")
    print("=" * 60)

    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    status = result.get("status", "unknown")

    if status == "succeeded":
        # -------------------------------------------------------------------------
        # 6a. Display success metrics
        # -------------------------------------------------------------------------
        best_score = result.get("best_score", 0.0)
        print("\n[6a] Optimization succeeded!")
        print("  Status: SUCCEEDED")
        print(f"  Best Score: {best_score:.2%}")
        print(f"  Duration: {format_duration(duration)}")

        # -------------------------------------------------------------------------
        # 6b. Download and preview the optimized graph
        # -------------------------------------------------------------------------
        print("\n[6b] Downloading optimized graph...")
        try:
            graph_txt = job.download_graph_txt()
            print("\n  Graph specification (first 500 chars):")
            print("  " + "-" * 56)
            preview = graph_txt[:500] + "..." if len(graph_txt) > 500 else graph_txt
            for line in preview.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 56)
        except Exception as e:
            print(f"  Could not download graph: {e}")

        # -------------------------------------------------------------------------
        # 6c. Run inference with the optimized graph
        # -------------------------------------------------------------------------
        print("\n[6c] Running inference example...")
        try:
            example_input = {"query": "How do I activate my card?"}
            output = job.run_inference(example_input)
            print(f"  Input:  {example_input}")
            print(f"  Output: {output}")
        except Exception as e:
            print(f"  Could not run inference: {e}")

    else:
        # -------------------------------------------------------------------------
        # Handle failure
        # -------------------------------------------------------------------------
        error = result.get("error", "Unknown error")
        print(f"\n  Status: {status.upper()}")
        print(f"  Error: {error}")
        print(f"  Duration: {format_duration(duration)}")


# ==============================================================================
# MAIN: RUN THE COMPLETE DEMO
# ==============================================================================


def main():
    """Run the complete GraphGen demo end-to-end."""

    print("\n" + "=" * 60)
    print("GRAPHGEN DEMO: Banking77 Intent Classification")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Configure backend
    backend_url = setup_backend()

    # Step 2: Configure API key
    api_key = setup_api_key()

    # Step 3: Build dataset
    dataset, label_names = build_banking77_graphgen_dataset(num_train_tasks=50, num_test_tasks=20)

    # Step 4: Create GraphGen job
    job = create_graphgen_job(
        dataset=dataset,
        label_names=label_names,
        backend_url=backend_url,
        api_key=api_key,
    )

    # Step 5: Submit and monitor
    result, optimization_duration = submit_and_monitor(job)

    # Step 6: Display results
    display_results(job, result, optimization_duration)

    # Final summary
    total_duration = time.time() - total_start
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"  Optimization time: {optimization_duration:.1f}s")
    print(f"  Total time: {total_duration:.1f}s")
    print("\n")


if __name__ == "__main__":
    main()
