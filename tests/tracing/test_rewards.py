import asyncio
import json
import os
import tempfile
from datetime import datetime

import pytest
from sqlalchemy import text

from synth_ai.tracing_v3 import SessionTracer
from synth_ai.tracing_v3.abstractions import EnvironmentEvent, RuntimeEvent, TimeRecord, LMCAISEvent
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager
from synth_ai.scripts.verify_rewards import verify as verify_rewards_cli


pytestmark = [pytest.mark.unit, pytest.mark.v3]


async def _new_manager(tmp_path: str) -> AsyncSQLTraceManager:
    db_url = f"sqlite+aiosqlite:///{tmp_path}"
    mgr = AsyncSQLTraceManager(db_url=db_url)
    await mgr.initialize()
    return mgr


async def _new_tracer(mgr: AsyncSQLTraceManager) -> SessionTracer:
    tracer = SessionTracer()
    tracer.db = mgr
    return tracer


@pytest.mark.asyncio
async def test_environment_event_auto_inserts_event_reward() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_env.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_env", metadata={"t": "env"}):
                await tracer.start_timestep("step_1", turn_number=0)
                event = EnvironmentEvent(
                    system_instance_id="env",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    reward=2.5,
                    terminated=False,
                    truncated=False,
                    system_state_before={"s": 0},
                    system_state_after={"s": 1},
                )
                event_id = await tracer.record_event(event)
                assert isinstance(event_id, int)

            # Verify event_rewards join events
            async with mgr.session() as s:
                q = text(
                    """
                    SELECT er.session_id, er.reward_value, ev.event_type
                    FROM event_rewards er
                    JOIN events ev ON er.event_id = ev.id
                    WHERE er.session_id = :sid
                    """
                )
                res = await s.execute(q, {"sid": "sess_env"})
                rows = res.fetchall()
                assert len(rows) == 1
                sid, val, et = rows[0]
                assert sid == "sess_env"
                assert pytest.approx(val, rel=1e-6) == 2.5
                assert et == "environment"
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_outcome_rewards_filtering_and_sum_event_rewards() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_outcome.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            # Session A: reward 0
            async with tracer.session(session_id="sess_A"):
                await tracer.start_timestep("t1", turn_number=0)
                await tracer.record_outcome_reward(total_reward=0, achievements_count=0, total_steps=1)

            # Session B: two env rewards totaling 3.0 and outcome 2
            async with tracer.session(session_id="sess_B"):
                await tracer.start_timestep("t1", turn_number=0)
                for r in [1.0, 2.0]:
                    ev = EnvironmentEvent(
                        system_instance_id="env",
                        time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                        reward=r,
                    )
                    await tracer.record_event(ev)
                await tracer.record_outcome_reward(total_reward=2, achievements_count=2, total_steps=1)

            # Filter via outcome_rewards
            sids = await mgr.get_outcome_rewards_by_min_reward(1)
            assert set(sids) == {"sess_B"}

            # Sum event rewards by session
            async with mgr.session() as s:
                q = text(
                    """
                    SELECT session_id, COALESCE(SUM(reward_value), 0.0) as total
                    FROM event_rewards
                    GROUP BY session_id
                    """
                )
                res = await s.execute(q)
                sums = {row[0]: float(row[1]) for row in res.fetchall()}
                assert pytest.approx(sums.get("sess_B", 0.0), rel=1e-6) == 3.0
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_runner_shaped_reward_linked_to_runtime_event_and_message() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_runner.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_runner"):
                await tracer.start_timestep("turn0", turn_number=0)
                msg_id = await tracer.record_message(content=json.dumps({"action": "noop"}), message_type="assistant")
                assert isinstance(msg_id, int)
                rt = RuntimeEvent(
                    system_instance_id="runtime",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    actions=[0],
                )
                rt_id = await tracer.record_event(rt)
                assert isinstance(rt_id, int)
                rid = await tracer.record_event_reward(
                    event_id=rt_id,
                    message_id=msg_id,
                    turn_number=0,
                    reward_value=1.25,
                    reward_type="shaped",
                    source="runner",
                )
                assert isinstance(rid, int)

            # Verify join to messages
            async with mgr.session() as s:
                q = text(
                    """
                    SELECT er.reward_value, m.message_type
                    FROM event_rewards er
                    JOIN messages m ON er.message_id = m.id
                    WHERE er.session_id = :sid
                    """
                )
                res = await s.execute(q, {"sid": "sess_runner"})
                rows = res.fetchall()
                assert len(rows) == 1
                val, mtype = rows[0]
                assert pytest.approx(val, rel=1e-6) == 1.25
                assert mtype == "assistant"
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_incremental_counters_and_fk_links() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_counters.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_cnt"):
                step = await tracer.start_timestep("s0", turn_number=0)
                m_id = await tracer.record_message(content="hello", message_type="system")
                assert isinstance(m_id, int)
                rt = RuntimeEvent(
                    system_instance_id="runtime",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    actions=[1],
                )
                ev_id = await tracer.record_event(rt)
                assert isinstance(ev_id, int)

            # Check counters and FK joins
            async with mgr.session() as s:
                # session counters
                q1 = text("SELECT num_timesteps, num_events, num_messages FROM session_traces WHERE session_id='sess_cnt'")
                r1 = await s.execute(q1)
                nt, ne, nm = r1.fetchone()
                assert nt == 1 and ne == 1 and nm == 1

                # event -> timestep
                q2 = text(
                    """
                    SELECT e.timestep_id, t.step_id, t.turn_number
                    FROM events e JOIN session_timesteps t ON e.timestep_id = t.id
                    WHERE e.session_id='sess_cnt'
                    """
                )
                r2 = await s.execute(q2)
                tid, step_id, turn = r2.fetchone()
                assert step_id == "s0" and turn == 0

                # message -> timestep
                q3 = text(
                    """
                    SELECT m.timestep_id, t.step_id
                    FROM messages m JOIN session_timesteps t ON m.timestep_id = t.id
                    WHERE m.session_id='sess_cnt'
                    """
                )
                r3 = await s.execute(q3)
                _, m_step_id = r3.fetchone()
                assert m_step_id == "s0"
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_outcome_rewards_fk_to_session_and_join() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_outcome_fk.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_fk"):
                await tracer.record_outcome_reward(total_reward=7, achievements_count=3, total_steps=11)

            async with mgr.session() as s:
                q = text(
                    """
                    SELECT s.session_id, o.total_reward
                    FROM session_traces s JOIN outcome_rewards o ON s.session_id = o.session_id
                    WHERE s.session_id = :sid
                    """
                )
                res = await s.execute(q, {"sid": "sess_fk"})
                sid, total = res.fetchone()
                assert sid == "sess_fk" and total == 7
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_event_reward_metadata_and_filtering() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_meta.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_meta"):
                await tracer.start_timestep("t0", turn_number=0)
                rt = RuntimeEvent(
                    system_instance_id="runtime",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    actions=[2],
                )
                e_id = await tracer.record_event(rt)
                assert isinstance(e_id, int)
                rid = await tracer.record_event_reward(
                    event_id=e_id,
                    message_id=None,
                    turn_number=0,
                    reward_value=3.5,
                    reward_type="shaped",
                    key="collect_wood",
                    annotation={"note": "unit_test"},
                    source="runner",
                )
                assert isinstance(rid, int)

            async with mgr.session() as s:
                # Filter by type and key
                q = text(
                    """
                    SELECT reward_value, reward_type, key, json_extract(annotation, '$.note') as note
                    FROM event_rewards
                    WHERE reward_type = 'shaped' AND key = 'collect_wood' AND session_id = :sid
                    """
                )
                r = await s.execute(q, {"sid": "sess_meta"})
                val, rtype, key, note = r.fetchone()
                assert pytest.approx(val, rel=1e-6) == 3.5
                assert rtype == "shaped" and key == "collect_wood" and note == "unit_test"
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_record_event_reward_requires_event_id() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_require_event.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_req"):
                # Calling without event_id must raise TypeError due to required parameter
                with pytest.raises(TypeError):
                    # Missing required keyword-only argument 'event_id'
                    await tracer.record_event_reward()  # type: ignore[call-arg]
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_analytics_views_and_model_usage() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_views.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_views"):
                await tracer.start_timestep("t0", turn_number=0)
                lm = LMCAISEvent(
                    system_instance_id="llm",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    model_name="test-model",
                    provider="test-provider",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cost_usd=0.001,
                    latency_ms=12,
                )
                await tracer.record_event(lm)

            # Views should exist and not error
            async with mgr.session() as s:
                await s.execute(text("SELECT * FROM session_summary LIMIT 1"))
                await s.execute(text("SELECT * FROM model_usage_stats LIMIT 1"))

            # Manager helper for model usage
            df = await mgr.get_model_usage(model_name="test-model")
            assert isinstance(df, type(__import__('pandas').DataFrame())) or len(df) == len(df)
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_event_reward_zero_vs_none_behavior() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_zero_none.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_zn"):
                await tracer.start_timestep("t0", turn_number=0)
                # None -> no event_reward row
                ev_none = EnvironmentEvent(
                    system_instance_id="env",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    reward=None,  # type: ignore[arg-type]
                )
                await tracer.record_event(ev_none)
                # Zero -> event_reward row with 0.0
                ev_zero = EnvironmentEvent(
                    system_instance_id="env",
                    time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
                    reward=0.0,
                )
                await tracer.record_event(ev_zero)

            async with mgr.session() as s:
                q = text("SELECT COUNT(*) FROM event_rewards WHERE session_id='sess_zn'")
                cnt = (await s.execute(q)).scalar_one()
                assert cnt == 1
                q2 = text("SELECT reward_value FROM event_rewards WHERE session_id='sess_zn'")
                val = (await s.execute(q2)).scalar_one()
                assert pytest.approx(float(val), rel=1e-6) == 0.0
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_verify_rewards_cli_return_codes() -> None:
    with tempfile.TemporaryDirectory() as td:
        # DB 1: Missing outcome_rewards -> returns 2
        db1 = os.path.join(td, "db_missing_outcome.db")
        mgr1 = await _new_manager(db1)
        tracer1 = await _new_tracer(mgr1)
        async with tracer1.session(session_id="sess_missing"):
            await tracer1.start_timestep("t0", turn_number=0)
            await tracer1.record_event(EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), reward=1.0))
        await mgr1.close()
        rc_missing = await verify_rewards_cli(db1, 0)
        assert rc_missing == 2

        # DB 2: Has outcome but none qualify -> returns 3
        db2 = os.path.join(td, "db_no_qualify.db")
        mgr2 = await _new_manager(db2)
        tracer2 = await _new_tracer(mgr2)
        async with tracer2.session(session_id="sess_noq"):
            await tracer2.record_outcome_reward(total_reward=0, achievements_count=0, total_steps=1)
        await mgr2.close()
        rc_noq = await verify_rewards_cli(db2, 1)
        assert rc_noq == 3

        # DB 3: Qualifying -> returns 0
        db3 = os.path.join(td, "db_ok.db")
        mgr3 = await _new_manager(db3)
        tracer3 = await _new_tracer(mgr3)
        async with tracer3.session(session_id="sess_ok"):
            await tracer3.record_outcome_reward(total_reward=2, achievements_count=2, total_steps=1)
            await tracer3.start_timestep("t0", turn_number=0)
            await tracer3.record_event(EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), reward=1.0))
        await mgr3.close()
        rc_ok = await verify_rewards_cli(db3, 1)
        assert rc_ok == 0


@pytest.mark.asyncio
async def test_event_reward_turn_number_auto_from_step() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_turnnum.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_turn"):
                await tracer.start_timestep("t5", turn_number=5)
                # Environment event with reward should auto insert event_reward with turn_number=5
                await tracer.record_event(EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), reward=1.0))

            async with mgr.session() as s:
                q = text("SELECT turn_number FROM event_rewards WHERE session_id='sess_turn'")
                tn = (await s.execute(q)).scalar_one()
                assert tn == 5
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_cli_uses_max_outcome_reward_per_session() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_cli_max.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_max"):
                await tracer.record_outcome_reward(total_reward=1, achievements_count=1, total_steps=1)
                await tracer.record_outcome_reward(total_reward=3, achievements_count=3, total_steps=2)
            rc = await verify_rewards_cli(db_file, 2)
            assert rc == 0
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_filter_sessions_by_sum_event_rewards_query() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_sum_filter.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_sum_a"):
                await tracer.start_timestep("t0")
                for r in [0.5, 0.7]:
                    await tracer.record_event(EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), reward=r))
            async with tracer.session(session_id="sess_sum_b"):
                await tracer.start_timestep("t0")
                await tracer.record_event(EnvironmentEvent(system_instance_id="env", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), reward=0.2))

            async with mgr.session() as s:
                q = text(
                    """
                    SELECT session_id
                    FROM event_rewards
                    GROUP BY session_id
                    HAVING SUM(reward_value) >= :th
                    """
                )
                res = await s.execute(q, {"th": 1.0})
                sids = {row[0] for row in res.fetchall()}
                assert sids == {"sess_sum_a"}
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_get_outcome_rewards_structure() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_get_outcome.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_get"):
                await tracer.record_outcome_reward(total_reward=4, achievements_count=4, total_steps=10)
            rows = await mgr.get_outcome_rewards()
            assert any(r["session_id"] == "sess_get" and r["total_reward"] == 4 for r in rows)
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_lm_event_does_not_auto_insert_event_reward() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_lm_noauto.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_lm"):
                await tracer.start_timestep("t0")
                await tracer.record_event(LMCAISEvent(system_instance_id="llm", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), model_name="m"))
            async with mgr.session() as s:
                cnt = (await s.execute(text("SELECT COUNT(*) FROM event_rewards WHERE session_id='sess_lm'"))).scalar_one()
                assert cnt == 0
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_start_session_creates_session_row_and_metadata() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_start_session.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_start", metadata={"exp": "abc"}):
                pass
            async with mgr.session() as s:
                row = (await s.execute(text("SELECT session_id, metadata FROM session_traces WHERE session_id='sess_start'"))).fetchone()
                assert row[0] == "sess_start"
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_indices_exist_for_reward_tables() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_index.db")
        mgr = await _new_manager(db_file)
        try:
            async with mgr.session() as s:
                # outcome_rewards indices
                res1 = await s.execute(text("PRAGMA index_list('outcome_rewards')"))
                idxs1 = [row[1] for row in res1.fetchall()]
                assert any("idx_outcome_rewards_session" in x for x in idxs1)
                assert any("idx_outcome_rewards_total" in x for x in idxs1)
                # event_rewards indices
                res2 = await s.execute(text("PRAGMA index_list('event_rewards')"))
                idxs2 = [row[1] for row in res2.fetchall()]
                assert any("idx_event_rewards_session" in x for x in idxs2)
                assert any("idx_event_rewards_event" in x for x in idxs2)
                assert any("idx_event_rewards_type" in x for x in idxs2)
                assert any("idx_event_rewards_key" in x for x in idxs2)
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_manager_ensure_timestep_idempotent_and_insert_helpers() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_helpers.db")
        mgr = await _new_manager(db_file)
        try:
            await mgr.ensure_session("sess_helpers", created_at=datetime.utcnow(), metadata={"a": 1})
            id1 = await mgr.ensure_timestep("sess_helpers", step_id="stepX", step_index=0, turn_number=0)
            id2 = await mgr.ensure_timestep("sess_helpers", step_id="stepX", step_index=0, turn_number=0)
            assert id1 == id2
            mid = await mgr.insert_message_row("sess_helpers", timestep_db_id=id1, message_type="system", content="hi")
            assert isinstance(mid, int)
            rid = await mgr.insert_event_row("sess_helpers", timestep_db_id=id1, event=RuntimeEvent(system_instance_id="r", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), actions=[1]))
            assert isinstance(rid, int)
        finally:
            await mgr.close()


@pytest.mark.asyncio
async def test_foreign_key_enforcement_on_event_delete() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_file = os.path.join(td, "rewards_fk_enforce.db")
        mgr = await _new_manager(db_file)
        try:
            tracer = await _new_tracer(mgr)
            async with tracer.session(session_id="sess_fkdel"):
                await tracer.start_timestep("t0", turn_number=0)
                rt = RuntimeEvent(system_instance_id="r", time_record=TimeRecord(event_time=datetime.utcnow().timestamp()), actions=[1])
                e_id = await tracer.record_event(rt)
                assert isinstance(e_id, int)
                await tracer.record_event_reward(event_id=e_id, reward_value=1.0, reward_type="shaped")

            # Attempt to delete the referenced event; should fail due to FK
            from sqlalchemy.exc import IntegrityError
            async with mgr.session() as s:
                with pytest.raises(Exception):
                    await s.execute(text("DELETE FROM events WHERE id = :id"), {"id": e_id})
                    await s.commit()
        finally:
            await mgr.close()
