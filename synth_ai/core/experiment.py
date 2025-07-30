from typing import List


class ExperimentalSystem:
    system_id: str
    system_version_id: str
    pass


class Experiment:
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    related_systems: List[ExperimentalSystem]
