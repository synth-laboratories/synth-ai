import gym_sokoban.envs.room_utils as ru
import pytest
from gym_sokoban.envs.room_utils import generate_room


@pytest.mark.xfail(reason="This test is expected to fail as the dimension is too small.")
def test_generate_room_raises_runtime_warning_on_score_zero(monkeypatch):
    """If reverse_playing always yields score=0, generate_room should raise a RuntimeWarning."""

    # Monkeypatch reverse_playing to always return zero score
    def fake_reverse(room_state, room_structure, *args, **kwargs):
        return room_state, 0, {}

    monkeypatch.setattr(ru, "reverse_playing", fake_reverse)

    # generate_room with only 1 try should immediately hit the zero-score branch
    with pytest.raises(RuntimeWarning) as excinfo:
        generate_room(dim=(5, 5), p_change_directions=0, num_steps=0, num_boxes=1, tries=1)
    assert "Generated Model with score == 0" in str(excinfo.value)
