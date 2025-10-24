class System:
    id: str
    name: str
    description: str
    pass


class SystemVersion:
    id: str
    system_id: str
    branch: str
    commit: str
    created_at: str
    description: str
    pass
