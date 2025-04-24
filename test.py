from typing import Any, Dict, List, Literal, Union, Optional

from pydantic import BaseModel, Field

from zyk import LM
from exp import make_openai_compatible

class ClusterAnalysis(BaseModel):
    summary: str
    impact_assessment: str
    recommendations: List[str]


class ClusterInstanceAnalysis(BaseModel):
    summary: str
    failure_mode_match: str
    evidence: List[str]


class EnrichmentAnalysis(BaseModel):
    summary: str
    quality_assessment: str
    improvement_suggestions: List[str]


class TraceAnalysis(BaseModel):
    summary: str
    key_events: List[str]
    failure_patterns: List[str]
    recommendations: List[str]


class AnalyzeToolParams(BaseModel):
    """Parameters for the analyze tool"""

    tool_type: Literal["analyze"]
    target_type: str = Field(
        description="Type of target to analyze",
        enum=["trace", "enrichment", "cluster", "cluster_instance"],
    )
    target_id: str = Field(description="UUID of the target to analyze")
    system_id: str = Field(description="System ID to analyze in")


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


class SimilaritySearchParams(BaseModel):
    tool_type: Literal["similarity_search"]
    query: str
    target: str
    system_id: str
    top_k: int


class ColumnUpdate(BaseModel):
    """Represents a single column update"""

    column: str = Field(description="Name of the column to update")
    value: str = Field(description="New value to set for the column")


class UpdateToolParams(BaseModel):
    """Parameters for the update tool"""

    target_type: str = Field(
        description="Type of target to update", enum=["cluster", "enrichment"]
    )
    target_id: str = Field(description="UUID of the target to update")
    updates: List[ColumnUpdate] = Field(description="List of column updates to apply")
    system_id: str = Field(description="System ID to update in")


class ToolCallArguments(BaseModel):
    tool_name: str
    #params: SimilaritySearchParams | UpdateToolParams | AnalyzeToolParams | ShowToolParams
    params: Union[
        SimilaritySearchParams,
        UpdateToolParams,
        AnalyzeToolParams,
        ShowToolParams,
    ]# = Field(discriminator="tool_type")



class ResponseModel(BaseModel):
    state_delta_instructions: str#Dict[str,str]#[str, Any]
    continue_execution: bool
    reasoning: str
    tool_calls: List[ToolCallArguments]


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "\n# Objective\nReview and analyze failure clusters to improve system reliability by:\n1. Consolidating duplicate or similar clusters\n2. Creating additional failure cluster instances where appropriate\n3. Marking inactive clusters\n4. Gathering facts and insights as specified in engineer instructions\n\n# CRITICAL RULE\nYOU MUST USE EXACTLY ONE TOOL IN EVERY SINGLE TURN. NO EXCEPTIONS.\nIf you don't use a tool, your turn will fail.\n\n# Important Rules\n1. You MUST use exactly one tool in each step - this is non-negotiable\n2. You MUST provide clear reasoning for your decisions\n3. You MUST document important findings and patterns\n4. You MUST focus on actionable improvements\n5. You MUST consider the impact of your changes on the system\n6. You MUST verify cluster relationships before making updates\n7. You MUST use similarity_search first to find valid UUIDs before using analyze or update\n8. You MUST only use valid UUID strings (e.g. \"123e4567-e89b-12d3-a456-426614174000\") when calling tools\n9. You MUST update final_results in your state with a comprehensive analysis before your last step\n\n# Analysis Process\n1. Start by using similarity_search to find relevant clusters and their instances\n2. Use analyze with specific UUIDs found from similarity_search to review clusters in detail\n3. For each cluster found:\n   - Use similarity_search to find similar clusters by searching annotations and failure modes\n   - Use analyze with specific UUIDs to check for duplicates\n   - Use similarity_search to find related failure instances\n4. Use save_notes to document findings and reasoning\n5. Use update with specific UUIDs to modify clusters\n6. Use analyze with specific UUIDs to verify changes\n7. Address any specific questions from engineer instructions\n\n# Quality Checklist\nBefore submitting your response, verify:\n\nContent Quality:\n[ ] All cluster references include specific UUIDs\n[ ] Recommendations include concrete implementation steps\n[ ] Success criteria are quantifiable\n[ ] Related clusters are explicitly linked\n[ ] Environmental factors are documented\n\nCompleteness:\n[ ] All required fields are populated\n[ ] No generic/vague descriptions\n[ ] All findings have supporting evidence\n[ ] All recommendations have clear next steps\n[ ] All systemic issues have root cause analysis\n\nSpecificity:\n[ ] Used specific examples for each finding\n[ ] Included measurable metrics\n[ ] Provided clear implementation steps\n[ ] Documented specific test scenarios\n[ ] Listed concrete success criteria\n[ ] ALL info in the final results are concrete and draw on hard facts\n\nRemember: You MUST use exactly one tool every single turn. If you don't use a tool, your turn will fail.\n\n\n# Cluster IDs You Are To Review and Potentially Update\nYou are specifically tasked with reviewing and modifying these clusters: 92470959-2dad-4a2b-8ea4-832c9b79fd28, 939ee5eb-88d3-430a-b5ed-878da80ee6ba, e7951231-3f68-440b-807a-a321d1db3ca5\nYou may search for other clusters and update them if appropriate. But the primary objective is to review the above clusters. Use the ShowTool to show their contents if necessary.\n\n\n# Step Budget Status\nYou have 5 steps remaining out of 6.\nIMPORTANT: Before your last step, make sure to update final_results in your state with a comprehensive analysis.\n\n\nEngineer's Custom Instructions:\n\n    Please analyze these clusters with the following focus:\n    1. Look for any clusters that might be duplicates based on similar failure modes\n    2. Check if we're missing any failure instances by comparing with similar traces\n    3. For any cluster marked as high priority, verify if it's still relevant\n    4. Pay special attention to clusters related to timeout errors\n    5. Provide specific recommendations for each cluster that needs changes\n    \n    Additional Questions:\n    - Are there any patterns in the failure modes that suggest a systemic issue?\n    - Which clusters would benefit from additional test cases?\n\n    Constraints: Please be extremely specific and concrete in your final-result recommendations and analyses.\n\n    Investigate in the first 2/3rds of your step budget, and then prepare to apply db changes in the last 1/3rd. Ensure you submit db changes.\n    Apply db changes by invoking the 'update' tool. It is absolute essential that you invoke this tool to apply the changes. Any information you include in final_results will be a helpful reference to the engineer, but actual changes must be applied using the 'update' tool.\n    \n# Available Tools\n- similarity_search: Search through traces, enrichments, clusters, and cluster instances using semantic similarity to the query. The tool works by embedding the query and each item, then calculating cosine similarity. Do not attempt to use keywords or advanced search operators.\n  - query: The search query to find similar items\n  - target: Target type to search in (one of: trace, enrichment, cluster, cluster_instance)\n  - system_id: System ID to search in\n  - top_k: Number of top results to return\n- update: Propose updates to clusters and enrichments\n  - target_type: Type of target to update (one of: cluster, enrichment)\n  - target_id: UUID of the target to update\n  - updates: List of column updates to apply\n  - system_id: System ID to update in\n- analyze: Analyze a specific trace, enrichment, cluster, or cluster instance\n  - target_type: Type of target to analyze (one of: trace, enrichment, cluster, cluster_instance)\n  - target_id: UUID of the target to analyze\n  - system_id: System ID to analyze in\n- show: Display database objects in a readable format\n  - requests: List of show requests\n  - system_id: System ID to query in",
        },
        {
            "role": "user",
            "content": "<objective>Review and manage failure clusters</objective>\n<plan>Begin by analyzing the provided failure clusters</plan>",
        }
    ]
    lm = LM(
        model_name="gpt-4o",
        formatting_model_name="gpt-4o-mini",
        temperature=0.1,
        structured_output_mode="forced_json",
    )
    response = lm.respond_sync(
        messages=messages,
        response_model=ResponseModel,
    )
    print(response)
