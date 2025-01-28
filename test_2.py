import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from zyk import LM

# Silence backoff logger
logging.getLogger("backoff").setLevel(logging.ERROR)


# Response models for LLM analysis
class ClusterAnalysis(BaseModel):
    """Analysis of a failure cluster"""

    summary: str
    impact_assessment: str
    recommendations: List[str]


class TraceAnalysis(BaseModel):
    """Analysis of a trace"""

    summary: str
    key_events: List[str]
    failure_patterns: List[str]
    recommendations: List[str]


class EnrichmentAnalysis(BaseModel):
    """Analysis of an enrichment"""

    summary: str
    quality_assessment: str
    improvement_suggestions: List[str]


class ClusterInstanceAnalysis(BaseModel):
    """Analysis of a cluster instance"""

    summary: str
    failure_mode_match: str
    evidence: List[str]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SearchResult(BaseModel):
    """A single search result with its similarity score"""

    id: str
    type: str
    content: Dict[str, Any]
    similarity_score: float


class SimilaritySearchParams(BaseModel):
    """Parameters for similarity search"""

    tool_type: Literal["similarity_search"]
    query: str = Field(description="The search query to find similar items")
    target: str = Field(
        description="Target type to search in",
        enum=["trace", "enrichment", "cluster", "cluster_instance"],
    )
    system_id: str = Field(description="System ID to search in")
    top_k: int = Field(description="Number of top results to return")  # default 5


@dataclass
class AnalysisChunk:
    """A chunk of data to be analyzed"""

    chunk_type: str  # e.g. "cluster", "instance", "trace", "enrichment"
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None



def chunk_data(
    data: Dict[str, Any],
    target_type: str,
    max_tokens: int = 100000,
    max_chars: int = 1000000,
) -> List[AnalysisChunk]:
    """Split data into chunks that fit within token and character limits."""
    # print(f"\n=== Starting chunk_data for {target_type} ===")
    # print(f"Initial max_tokens: {max_tokens}, max_chars: {max_chars}")

    chunks = []

    if target_type == "cluster":
        # First chunk is the cluster metadata (always keep together)
        cluster_meta = {
            "id": data["id"],
            "annotation": data["annotation"],
            "failure_mode": data["failure_mode"],
            "success_mode": data["success_mode"],
            "priority": data["priority"],
            "actionable": data["actionable"],
        }
        chunks.append(AnalysisChunk("cluster_meta", cluster_meta))
        # print("\nCreated cluster_meta chunk")

        # For each instance, create separate chunks that align with _format_data_for_analysis
        for idx, instance in enumerate(data.get("instances", []), 1):
            # print(
            #     f"\n--- Processing Instance {idx}/{len(data.get('instances', []))} ---"
            # )

            # Instance metadata chunk (always keep together)
            instance_meta = {
                "id": instance["id"],
                "type": instance["type"],
                "annotation": instance["annotation"],
                "failure_annotation": instance["failure_annotation"],
                "event_indicators": instance["event_indicators"],
            }
            chunks.append(AnalysisChunk("instance_meta", instance_meta))
            # print("Created instance_meta chunk")

            # Handle trace data - use 75% of limits for better LLM context
            if instance.get("trace") and instance["trace"].get("data"):
                trace_data = instance["trace"]["data"]
                if isinstance(trace_data, dict):
                    trace_data = json.dumps(trace_data, indent=2)

                if isinstance(trace_data, str) and trace_data.strip():
                    # Use 75% of limits for trace data
                    trace_tokens = int(max_tokens * 0.75)
                    trace_chars = int(max_chars * 0.75)
                    # print(
                    #     f"Creating trace chunk with limits: tokens={trace_tokens}, chars={trace_chars}"
                    # )

                    chunks.append(
                        AnalysisChunk(
                            "trace_data",
                            {"instance_id": instance["id"], "data": trace_data},
                        )
                    )

            # Handle enrichments - keep each enrichment as a single chunk
            if instance.get("enrichments"):
                # p  # rint(f"Processing {len(instance['enrichments'])} enrichments")
                for enr_idx, enrichment in enumerate(instance["enrichments"], 1):
                    chunks.append(
                        AnalysisChunk(
                            "enrichment",
                            {
                                "instance_id": instance["id"],
                                "description": enrichment["description"],
                                "annotation": enrichment["annotation"],
                                "rating": enrichment["rating"],
                                "confidence": enrichment["confidence"],
                                "reward": enrichment["reward"],
                            },
                        )
                    )

    elif target_type in ["trace", "enrichment", "cluster_instance"]:
        # For other types, create appropriate chunks based on _format_data_for_analysis
        meta = {k: v for k, v in data.items() if k != "trace" and k != "enrichments"}
        chunks.append(AnalysisChunk(f"{target_type}_meta", meta))

        # Handle trace data if present
        if data.get("trace") and data["trace"].get("data"):
            trace_data = data["trace"]["data"]
            if isinstance(trace_data, dict):
                trace_data = json.dumps(trace_data, indent=2)

            if isinstance(trace_data, str) and trace_data.strip():
                trace_tokens = int(max_tokens * 0.75)
                trace_chars = int(max_chars * 0.75)
                chunks.append(AnalysisChunk("trace_data", {"data": trace_data}))

        # Handle enrichments if present
        if data.get("enrichments"):
            for enrichment in data["enrichments"]:
                chunks.append(AnalysisChunk("enrichment", enrichment))

    # Now split any chunks that exceed max tokens/chars
    # print("\n=== Processing final chunks ===")
    final_chunks = []
    for chunk_idx, chunk in enumerate(chunks, 1):
        # print(f"\nProcessing chunk {chunk_idx}/{len(chunks)} ({chunk.chunk_type})")

        # Calculate effective limits based on chunk type
        effective_max_tokens = max_tokens
        effective_max_chars = max_chars
        if chunk.chunk_type == "trace_data":
            effective_max_tokens = int(max_tokens * 0.75)
            effective_max_chars = int(max_chars * 0.75)
            # print(
            #     f"Using trace data limits: tokens={effective_max_tokens}, chars={effective_max_chars}"
            # )

        split_chunks = split_chunk_by_tokens(
            chunk, max_tokens=effective_max_tokens, max_chars=effective_max_chars
        )
        # print(f"Split into {len(split_chunks)} sub-chunks")
        final_chunks.extend(split_chunks)

    # Print final statistics
    # print("\n=== Final Chunk Statistics ===")
    chunk_types = {}
    for chunk in final_chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        formatted_data = _format_chunk_data(chunk)
        token_count = count_tokens(formatted_data)
        char_count = len(formatted_data)
        # print(
        #     f"Chunk ({chunk.chunk_type}) token length: {token_count}, char length: {char_count}"
        # )

    # print("\nChunk type distribution:")
    # for chunk_type, count in chunk_types.items():
    #     print(f"{chunk_type}: {count} chunks")

    return final_chunks


async def analyze_chunk(
    lm: LM,
    chunk: AnalysisChunk,
    system_message: str,
    response_model: Any,
) -> Dict[str, Any]:
    """Analyze a single chunk of data"""
    formatted_data = _format_chunk_data(chunk)
    if "gemini" not in lm.model_name:
        try:
            response = await lm.respond_async(
                system_message=system_message,
                user_message=formatted_data,
                response_model=response_model,
            )
        except:
            print("Falling back to gemini")
            gemini_lm = LM(
                model_name="gemini-1.5-flash",
                formatting_model_name="gpt-4o-mini",
                temperature=0.1,
            )
            response = await gemini_lm.respond_async(
                system_message=system_message,
                user_message=formatted_data,
                response_model=response_model,
            )
    else:
        response = await lm.respond_async(
            system_message=system_message,
            user_message=formatted_data,
            response_model=response_model,
        )

    return response.dict()


def _format_chunk_data(chunk: AnalysisChunk) -> str:
    """Format a chunk for analysis"""
    if chunk.chunk_type == "cluster_meta":
        return (
            f"Analyze this failure cluster metadata:\n\n"
            f"Failure Mode: {chunk.data['failure_mode']}\n"
            f"Success Mode: {chunk.data['success_mode']}\n"
            f"Current Priority: {chunk.data['priority']}\n"
            f"Actionable Status: {chunk.data['actionable']}\n"
            f"Detailed Annotation: {chunk.data['annotation']}\n"
        )
    elif chunk.chunk_type == "instance":
        return (
            f"Analyze this cluster instance:\n\n"
            f"Type: {chunk.data['type']}\n"
            f"Annotation: {chunk.data['annotation']}\n"
            f"Failure Annotation: {chunk.data['failure_annotation']}\n"
            f"Event Indicators: {chunk.data['event_indicators']}\n"
            f"\nTrace Data:\n{chunk.data.get('trace', {}).get('data', 'No trace data')}\n"
            + (
                "\nTrace Enrichments:\n"
                + "\n".join(
                    f"- Description: {e['description']}\n"
                    f"  Annotation: {e['annotation']}\n"
                    f"  Rating: {e['rating']}\n"
                    f"  Confidence: {e['confidence']}\n"
                    f"  Reward: {e['reward']}"
                    for e in chunk.data.get("enrichments", [])
                )
                if chunk.data.get("enrichments")
                else ""
            )
        )
    elif chunk.chunk_type in ["trace", "trace_data"]:
        return f"Analyze this trace data:\n\n{chunk.data.get('data', '')}"
    elif chunk.chunk_type == "enrichment":
        return (
            f"Analyze this trace enrichment:\n\n"
            f"Description: {chunk.data['description']}\n"
            f"Annotation: {chunk.data['annotation']}\n"
            f"Rating: {chunk.data['rating']}\n"
            f"Confidence: {chunk.data['confidence']}\n"
            f"Reward: {chunk.data['reward']}\n"
        )
    return str(chunk.data)


class ConsolidatedAnalysis(BaseModel):
    """Consolidated analysis from multiple chunks"""

    summary: str
    key_findings: List[str]
    recommendations: List[str]

# Model token and character limits
MODEL_LIMITS = {
    "gpt-4o-mini": {"max_tokens": 50_000, "max_chars": 100_000},
    "gemini-1.5-flash": {"max_tokens": 500_000, "max_chars": 2_000_000},
}
class AnalyzeToolParams(BaseModel):
    """Parameters for the analyze tool"""

    tool_type: Literal["analyze"]
    target_type: str = Field(
        description="Type of target to analyze",
        enum=["trace", "enrichment", "cluster", "cluster_instance"],
    )
    target_id: str = Field(description="UUID of the target to analyze")
    system_id: str = Field(description="System ID to analyze in")
    model_name: str = Field(
        description="Model to use for analysis",
        default="gpt-4o-mini",
        enum=list(MODEL_LIMITS.keys()),
    )


class ColumnUpdate(BaseModel):
    """Represents a single column update"""

    column: str = Field(description="Name of the column to update")
    value: str = Field(description="New value to set for the column")


class MergeUpdate(BaseModel):
    """Represents a cluster merge operation"""

    source_cluster_ids: List[str] = Field(
        description="List of UUIDs of clusters to merge from"
    )
    target_cluster_id: str = Field(
        description="UUID of the target cluster to merge into. If not provided, a new cluster will be created",
        default=None,
    )
    instance_ids: List[str] = Field(
        description="List of instance UUIDs to include in the merge"
    )
    merge_reason: str = Field(
        description="Explanation of why these clusters should be merged"
    )


class UpdateToolParams(BaseModel):
    """Parameters for the update tool"""

    tool_type: Literal["update"]
    target_type: str = Field(
        description="Type of target to update", enum=["cluster", "enrichment"]
    )
    target_id: str = Field(description="UUID of the target to update")
    column_updates: List[ColumnUpdate] = Field(
        description="List of column updates to apply", default_factory=list
    )
    merge_updates: List[MergeUpdate] = Field(
        description="List of cluster merges to perform", default_factory=list
    )
    system_id: str = Field(description="System ID to update in")


# class UpdateToolResponse(BaseModel):
#     """Response model for the update tool"""

#     status: str = Field(description="Status of the update operation (success/error)")
#     valid_updates: List[Dict[str, str]] = Field(
#         description="List of updates that were validated and stored",
#         default_factory=list,
#     )
#     invalid_updates: List[Dict[str, Any]] = Field(
#         description="List of updates that failed validation with reasons",
#         default_factory=list,
#     )
#     target_id: str = Field(description="ID of the target that was updated")


class SaveNotesParams(BaseModel):
    """Parameters for the save notes tool"""

    tool_type: Literal["save_notes"]
    notes: str = Field(description="Notes to save")


class SaveNotesToolResponse(BaseModel):
    """Response model for the save notes tool"""

    status: str = Field(description="Status of the save operation (success/error)")
    notes: str = Field(description="The notes that were saved")


class CheckNotesParams(BaseModel):
    """Parameters for checking saved notes"""

    tool_type: Literal["check_notes"]
    pass


class CheckNotesToolResponse(BaseModel):
    """Response model for checking saved notes"""

    notes: List[str] = Field(
        description="List of previously saved notes", default_factory=list
    )


class ShowRequest(BaseModel):
    """Request to show a database object"""

    table_name: str = Field(
        description="Name of the table to query",
        enum=["FailureCluster", "FailureClusterInstance", "Trace", "TraceEnrichment"],
    )
    where_column: str = Field(description="Column name for the WHERE clause")
    where_value: str = Field(description="Value for the WHERE clause")
    columns: List[str] = Field(description="List of columns to display")


class ShowToolParams(BaseModel):
    """Parameters for the show tool"""

    tool_type: Literal["show"]
    requests: List[ShowRequest] = Field(description="List of show requests")
    system_id: str = Field(description="System ID to query in")


# class ShowToolResponse(BaseModel):
#     """Response model for showing database objects"""

#     results: List[Dict[str, Any]] = Field(
#         description="List of results from show requests", default_factory=list
class ToolCallResponse(BaseModel):
    """Response from the LLM containing a tool call"""

    tool_name: str
    arguments: Union[
        SimilaritySearchParams,
        UpdateToolParams,
        AnalyzeToolParams,
        ShowToolParams,
    ]  # = Field(discriminator="tool_type")
    # ArgumentTypes = Field(discriminator="tool_type")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary format that's compatible with the ACI interface"""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }

class LLMResponse(BaseModel):
    """Structured response from the LLM"""

    state_delta_instructions: str  # Dict[str, Any]
    continue_execution: bool
    reasoning: str
    tool_calls: List[ToolCallResponse]

class EnhancedToolParams(BaseModel):
    """Base model for enhanced tool parameters that includes environment fields"""

    system_id: str = Field(default="")
    cluster_ids: List[str] = Field(default_factory=list)


system_message = """# Objective
Review and analyze failure clusters to improve system reliability by:
1. Consolidating duplicate or similar clusters
2. Creating additional failure cluster instances where appropriate
3. Marking inactive clusters
4. Gathering facts and insights as specified in engineer instructions

# CRITICAL RULE
YOU MUST USE EXACTLY ONE TOOL IN EVERY SINGLE TURN. NO EXCEPTIONS.
If you don't use a tool, your turn will fail.

# Important Rules
1. You MUST use exactly one tool in each step - this is non-negotiable
2. You MUST provide clear reasoning for your decisions
3. You MUST document important findings and patterns
4. You MUST focus on actionable improvements
5. You MUST consider the impact of your changes on the system
6. You MUST verify cluster relationships before making updates
7. You MUST use similarity_search first to find valid UUIDs before using analyze or update
8. You MUST only use valid UUID strings (e.g. "123e4567-e89b-12d3-a456-426614174000") when calling tools
9. You MUST update final_results in your state with a comprehensive analysis before your last step

# Analysis Process
1. Start by using similarity_search to find relevant clusters and their instances
2. Use analyze with specific UUIDs found from similarity_search to review clusters in detail
3. For each cluster found:
   - Use similarity_search to find similar clusters by searching annotations and failure modes
   - Use analyze with specific UUIDs to check for duplicates
   - Use similarity_search to find related failure instances

5. Use update with specific UUIDs to modify clusters
6. Use analyze with specific UUIDs to verify changes
7. Address any specific questions from engineer instructions

# Quality Checklist
Before submitting your response, verify:

Content Quality:
[ ] All cluster references include specific UUIDs
[ ] Recommendations include concrete implementation steps
[ ] Success criteria are quantifiable
[ ] Related clusters are explicitly linked
[ ] Environmental factors are documented

Completeness:
[ ] All required fields are populated
[ ] No generic/vague descriptions
[ ] All findings have supporting evidence
[ ] All recommendations have clear next steps
[ ] All systemic issues have root cause analysis

Specificity:
[ ] Used specific examples for each finding
[ ] Included measurable metrics
[ ] Provided clear implementation steps
[ ] Documented specific test scenarios
[ ] Listed concrete success criteria
[ ] ALL info in the final results are concrete and draw on hard facts

Remember: You MUST use exactly one tool every single turn. If you don't use a tool, your turn will fail.


# Cluster IDs You Are To Review and Potentially Update
You are specifically tasked with reviewing and modifying these clusters: 92470959-2dad-4a2b-8ea4-832c9b79fd28, 939ee5eb-88d3-430a-b5ed-878da80ee6ba, e7951231-3f68-440b-807a-a321d1db3ca5
You may search for other clusters and update them if appropriate. But the primary objective is to review the above clusters. Use the ShowTool to show their contents if necessary.


# Step Budget Status
You have 3 steps remaining out of 4.
IMPORTANT: Before your last step, make sure to update final_results in your state with a comprehensive analysis.


Engineer's Custom Instructions:

    Please analyze these clusters with the following focus:
    1. Look for any clusters that might be duplicates based on similar failure modes
    2. Check if we're missing any failure instances by comparing with similar traces
    3. For any cluster marked as high priority, verify if it's still relevant
    4. Pay special attention to clusters related to timeout errors
    5. Provide specific recommendations for each cluster that needs changes
    
    Additional Questions:
    - Are there any patterns in the failure modes that suggest a systemic issue?
    - Which clusters would benefit from additional test cases?

    Constraints: Please be extremely specific and concrete in your final-result recommendations and analyses.
    Never attempt to modify message history in the state. That is procedurally generated and any attempted edits will be ignored, leading only to lost time.

    Investigate in the first 2/3rds of your step budget, and then prepare to apply db changes in the last 1/3rd. Ensure you submit db changes. If you submit actions you are happy with, you may opt to set continue_execution to false.
    Updating columns is valuable, but it is especially valuable to *merge duplicate clusters* when you find them. If you find a duplicate, be sure the merge them. This is your highest priority objective.
    When merging clusters, strive to assign all cluster instances within those clusters to the surviving cluster. Only refrain from doing so if the merged cluster does not quite fit the instance.
    Apply db changes by invoking the 'update' tool. It is absolute essential that you invoke this tool to apply the changes.
    
# Available Tools
- similarity_search: Search through traces, enrichments, clusters, and cluster instances using semantic similarity to the query. The tool works by embedding the query and each item, then calculating cosine similarity. Do not attempt to use keywords or advanced search operators.
  - query: The search query to find similar items
  - target: Target type to search in (one of: trace, enrichment, cluster, cluster_instance)
  - system_id: System ID to search in
  - top_k: Number of top results to return
- show: Display database objects in a readable format
  - tool_type: 
  - requests: List of show requests
  - system_id: System ID to query in
- analyze: Analyze a specific trace, enrichment, cluster, or cluster instance
  - tool_type: 
  - target_type: Type of target to analyze (one of: trace, enrichment, cluster, cluster_instance)
  - target_id: UUID of the target to analyze
  - system_id: System ID to analyze in
  - model_name: Model to use for analysis (one of: gpt-4o-mini, gemini-1.5-flash)
- update: Propose updates to clusters and enrichments. 
            For clusters, valid columns are:
            - cluster_annotation (string)
            - cluster_failure_mode (string)
            - cluster_success_mode (string)
            - cluster_priority (must be: 'low', 'medium', 'high')
            - cluster_actionable (must be: 'generate_data', 'finetune', 'pull_request', 'no_action')
            
  - tool_type: 
  - target_type: Type of target to update (one of: cluster, enrichment)
  - target_id: UUID of the target to update
  - column_updates: List of column updates to apply
  - merge_updates: List of cluster merges to perform
  - system_id: System ID to update in"""
user_message = """<objective>No objective set</objective>
<plan>No plan set</plan>
<final_results>No results yet</final_results>"""

from zyk import LM

output = LM(
    model_name="claude-3-5-sonnet-20241022",
    formatting_model_name="gpt-4o-mini",
    temperature=0.1,
    structured_output_mode="forced_json",
).respond_sync(
    system_message=system_message,
    user_message=user_message,
    response_model=LLMResponse,
)
print(output)
