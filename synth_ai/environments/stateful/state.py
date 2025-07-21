from synth_ai.environments.v0_observability.log import EnvironmentStepRecord


class State:
    async def get_environment_step_record(self) -> EnvironmentStepRecord:
        """Return the latest environment step record."""
        raise NotImplementedError
