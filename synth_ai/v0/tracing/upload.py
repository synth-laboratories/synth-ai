# ruff: noqa
import asyncio
import json
import logging
import os
import ssl
import time
from typing import Any, Dict, List, Tuple, TypedDict

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT

from .abstractions import Dataset, SystemTrace
from .events.store import event_store

load_dotenv()


# NOTE: This may cause memory issues in the future
def validate_json(data: Dict[str, Any]) -> None:
    """Validate that a dictionary contains only JSON-serializable values.

    Args:
        data: Dictionary to validate for JSON serialization

    Raises:
        ValueError: If the dictionary contains non-serializable values
    """

    try:
        json.dumps(data)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"Contains non-JSON-serializable values: {e}. {data}")


def createPayload(dataset: Dataset, traces: List[SystemTrace]) -> Dict[str, Any]:
    payload = {
        "traces": [trace.to_dict() for trace in traces],  # Convert SystemTrace objects to dicts
        "dataset": dataset.to_dict(),
    }
    return payload


class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            ssl_context=ctx,
        )


def load_signed_url(signed_url: str, dataset: Dataset, traces: List[SystemTrace]) -> None:
    payload = createPayload(dataset, traces)
    validate_json(payload)

    session = requests.Session()
    adapter = TLSAdapter()
    session.mount("https://", adapter)

    try:
        response = session.put(
            signed_url, json=payload, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        print(f"Request payload: {payload}")  # Debugging info
        raise

    if response.status_code != 200:
        raise ValueError(
            f"Failed to load signed URL Status Code: {response.status_code} "
            f"Response: {response.text}, Signed URL: {signed_url}"
        )
    else:
        pass


def send_system_traces_s3(
    dataset: Dataset,
    traces: List[SystemTrace],
    base_url: str,
    api_key: str,
    system_id: str,
    system_name: str,
    verbose: bool = False,
):
    upload_id, signed_url = get_upload_id(base_url, api_key, system_id, system_name, verbose)
    load_signed_url(signed_url, dataset, traces)

    token_url = f"{base_url}/v1/auth/token"
    try:
        token_response = requests.get(token_url, headers={"customer_specific_api_key": api_key})
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error obtaining access token: {e}")
        raise

    api_url = f"{base_url}/v1/uploads/process-upload/{upload_id}"
    data = {"signed_url": signed_url}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()

        response_data = response.json()
        upload_id = response_data.get("upload_id")
        signed_url = response_data.get("signed_url")
        return upload_id, signed_url
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


def get_upload_id(
    base_url: str, api_key: str, system_id: str, system_name: str, verbose: bool = False
) -> Tuple[str, str]:
    """
    Modified client-side function to send both system_id and system_name.
    """
    token_url = f"{base_url}/v1/auth/token"
    token_response = requests.get(token_url, headers={"customer_specific_api_key": api_key})
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]

    # Include system_name in the query parameters
    api_url = (
        f"{base_url}/v1/uploads/get-upload-id-signed-url?"
        f"system_id={system_id}&system_name={system_name}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        upload_id = response.json()["upload_id"]
        signed_url = response.json()["signed_url"]
        return upload_id, signed_url
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


class UploadValidator(BaseModel):
    """Validator for upload data format."""

    model_config = ConfigDict(
        from_attributes=True,  # Replaces the deprecated orm_mode
        validate_assignment=True,
        extra="forbid",  # Prevent additional fields
    )

    traces: List[Dict[str, Any]]
    dataset: Dict[str, Any]

    @field_validator("traces")
    @classmethod
    def validate_traces(cls, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not traces:
            raise ValueError("Traces list cannot be empty")

        for trace in traces:
            # Validate required fields in each trace
            if "system_instance_id" not in trace:
                raise ValueError("Each trace must have a system_instance_id")
            if "partition" not in trace:
                raise ValueError("Each trace must have a partition")

            # Validate metadata if present
            if "metadata" in trace and trace["metadata"] is not None:
                if not isinstance(trace["metadata"], dict):
                    raise ValueError("Metadata must be a dictionary")

            # Validate partition structure
            partition = trace["partition"]
            if not isinstance(partition, list):
                raise ValueError("Partition must be a list")

            for part in partition:
                if "partition_index" not in part:
                    raise ValueError("Each partition element must have a partition_index")
                if "events" not in part:
                    raise ValueError("Each partition element must have an events list")

                # Validate events
                events = part["events"]
                if not isinstance(events, list):
                    raise ValueError("Events must be a list")

                for event in events:
                    required_fields = [
                        "event_type",
                        "opened",
                        "closed",
                        "partition_index",
                    ]
                    missing_fields = [f for f in required_fields if f not in event]
                    if missing_fields:
                        raise ValueError(f"Event missing required fields: {missing_fields}")

        return traces

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, dataset: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["questions", "reward_signals"]
        missing_fields = [f for f in required_fields if f not in dataset]
        if missing_fields:
            raise ValueError(f"Dataset missing required fields: {missing_fields}")

        # Validate questions
        questions = dataset["questions"]
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")

        for question in questions:
            if "intent" not in question or "criteria" not in question:
                raise ValueError("Each question must have intent and criteria")

        # Validate reward signals
        reward_signals = dataset["reward_signals"]
        if not isinstance(reward_signals, list):
            raise ValueError("Reward signals must be a list")

        return dataset


def validate_upload(traces: List[Dict[str, Any]], dataset: Dict[str, Any]):
    # Validate the upload format before sending to server.
    # Raises ValueError if validation fails.
    try:
        UploadValidator(traces=traces, dataset=dataset)
        return True
    except ValueError as e:
        raise ValueError(f"Upload validation failed: {str(e)}")


def is_event_loop_running():
    try:
        asyncio.get_running_loop()  # Check if there's a running event loop
        return True
    except RuntimeError:
        # This exception is raised if no event loop is running
        return False


def format_upload_output(
    dataset: Dataset, traces: List[SystemTrace]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Format questions array
    questions_data = [
        {"intent": q.intent, "criteria": q.criteria, "id": q.id} for q in dataset.questions
    ]

    # Format reward signals array with error handling
    reward_signals_data = [
        {
            "system_instance_id": rs.system_instance_id,
            "reward": rs.reward,
            "question_id": rs.question_id,
            "annotation": rs.annotation if hasattr(rs, "annotation") else None,
        }
        for rs in dataset.reward_signals
    ]

    # Format traces array
    traces_data = [
        {
            "system_instance_id": t.system_instance_id,
            "metadata": t.metadata if t.metadata else None,
            "partition": [
                {
                    "partition_index": p.partition_index,
                    "events": [e.to_dict() for e in p.events],
                }
                for p in t.partition
            ],
        }
        for t in traces
    ]

    return questions_data, reward_signals_data, traces_data


class UploadIdResponse(TypedDict):
    message: str
    upload_id: str
    signed_url: str


class ProcessUploadResponse(TypedDict):
    status: str
    upload_id: str
    signed_url: str


def upload(
    dataset: Dataset,
    traces: List[SystemTrace] = [],
    verbose: bool = False,
    show_payload: bool = False,
) -> Tuple[
    ProcessUploadResponse,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Upload all system traces and dataset to the server.

    Args:
        dataset: Dataset containing questions and reward signals
        traces: List of system traces to upload
        verbose: Whether to print verbose output
        show_payload: Whether to show the payload being sent

    Returns:
        Tuple containing:
        - response: Server response with status, upload_id, and signed_url
        - questions_json: List of formatted questions
        - reward_signals_json: List of formatted reward signals
        - traces_json: List of formatted traces

    Raises:
        ValueError: If no system traces found or validation fails
        requests.exceptions.HTTPError: If server request fails
        RuntimeError: If SYNTH_API_KEY environment variable not set
    """
    return upload_helper(dataset, traces, verbose, show_payload)


def upload_helper(
    dataset: Dataset,
    traces: List[SystemTrace] = [],
    verbose: bool = False,
    show_payload: bool = False,
) -> Tuple[
    ProcessUploadResponse,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Helper function to handle the upload process.

    Returns same type as upload() function.
    """
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY environment variable not set")
    base_url = os.getenv("SYNTH_ENDPOINT_OVERRIDE", PROD_BASE_URL_DEFAULT)

"""Legacy block below retained for reference and disabled for linting/parsing.
Start disabled block.
"""
"""
from .decorators import _local, active_events_var
from .trackers import synth_tracker_async, synth_tracker_sync

    # First close any tracker events
    if hasattr(synth_tracker_async, "active_events"):
        for event_type, event in list(synth_tracker_async.active_events.items()):
            if event and event.closed is None:
                event.closed = time.time()
                try:
                    event_store.add_event(
                        event.system_name,
                        event.system_id,
                        event.system_instance_id,
                        event,
                    )
                    if verbose:
                        print(f"Closed and stored tracker async event: {event_type}")
                except Exception as e:
                    logging.error(f"Failed to store tracker event {event_type}: {str(e)}")
        synth_tracker_async.active_events.clear()

    # End all active events before uploading
    if hasattr(_local, "active_events"):
        for event_type, event in _local.active_events.items():
            if event and event.closed is None:
                event.closed = time.time()
                if hasattr(_local, "system_instance_id"):
                    try:
                        event_store.add_event(
                            _local.system_name,
                            _local.system_id,
                            _local.system_instance_id,
                            event,
                        )
                        if verbose:
                            print(f"Closed and stored active event: {event_type}")
                    except Exception as e:
                        logging.error(f"Failed to store event {event_type}: {str(e)}")
        _local.active_events.clear()

    # NEW: Close all open asynchronous events
    active_events_async = active_events_var.get()
    if active_events_async:
        current_time = time.time()
        for event_type, event in list(active_events_async.items()):
            if event and event.closed is None:
                event.closed = current_time
                try:
                    event_store.add_event(
                        event.system_name,
                        event.system_id,
                        event.system_instance_id,
                        event,
                    )
                    if verbose:
                        print(f"Closed and stored async event: {event_type}")
                except Exception as e:
                    logging.error(f"Failed to store async event {event_type}: {str(e)}")
        active_events_var.set({})

    # Also close any unclosed events in existing traces
    logged_traces = event_store.get_system_traces()
    traces = logged_traces + traces
    # traces = event_store.get_system_traces() if len(traces) == 0 else traces
    current_time = time.time()
    for trace in traces:
        for partition in trace.partition:
            for event in partition.events:
                if event.closed is None:
                    event.closed = current_time
                    event_store.add_event(
                        trace.system_name,
                        trace.system_id,
                        trace.system_instance_id,
                        event,
                    )
                    if verbose:
                        print(f"Closed existing unclosed event: {event.event_type}")

    try:
        # Get traces and convert to dict format
        if len(traces) == 0:
            raise ValueError("No system traces found")
        traces_dict = [trace.to_dict() for trace in traces]
        dataset_dict = dataset.to_dict()

        # Validate upload format
        if verbose:
            print("Validating upload format...")
        validate_upload(traces_dict, dataset_dict)
        if verbose:
            print("Upload format validation successful")

        # Send to server
        upload_id, signed_url = send_system_traces_s3(
            dataset=dataset,
            traces=traces,
            base_url=base_url,
            api_key=api_key,
            system_id=traces[0].system_id,
            system_name=traces[0].system_name,
            verbose=verbose,
        )

        questions_json, reward_signals_json, traces_json = format_upload_output(dataset, traces)
        return (
            {
                "status": "success",
                "upload_id": upload_id,
                "signed_url": signed_url,
            },
            questions_json,
            reward_signals_json,
            traces_json,
        )

    except ValueError as e:
        if verbose:
            print("Validation error:", str(e))
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise
    except requests.exceptions.HTTPError as e:
        if verbose:
            print("HTTP error occurred:", e)
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise
"""
