"""Every print of an archive keeps its own record, start to end (archive redesign P0).

The archive redesign moves what describes one print -- status, times, plate,
failure reason, charge -- from the archive onto the run. These tests drive
real print starts and completions against the database and pin what each run
records today, so every step of the move is checked against the same outcome.
They assert on the print log first; what the archive row shows is pinned
where the archive page reads it today, and changes there are meant to be
made here on purpose.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from backend.app import main as main_module
from backend.app.main import register_expected_print
from backend.app.models.archive import PrintArchive
from backend.app.models.finance import CostCenter, WalletTransaction
from backend.app.models.print_log import PrintLogEntry
from backend.app.models.print_queue import PrintQueueItem
from backend.app.models.settings import Settings
from backend.app.models.user import User
from backend.tests.integration.test_external_print_confirmation_1898 import (  # noqa: F401
    SUBTASK,
    _clear_print_state,
    _drive_print_start,
)
from backend.tests.integration.test_print_complete_end_to_end import _complete as _complete_concurrently

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

FILE = f"{SUBTASK}.gcode.3mf"


async def _complete(test_engine, printer_id: int, status: str, name: str):
    """``on_print_complete`` with its background work run one task at a time.

    The SQLite test database is one shared connection, so tasks writing at
    the same time fail each other ("cannot commit transaction - SQL
    statements in progress"), which a real database does not. The finish
    photo is left out of the turn-taking: the notification task waits for it.
    """
    from unittest.mock import patch

    from backend.app.core.tasks import spawn_background_task

    turn = asyncio.Lock()

    async def in_turn(coro):
        async with turn:
            return await coro

    def spawn(coro, name=None, **kwargs):
        if name != "background-finish-photo":
            coro = in_turn(coro)
        return spawn_background_task(coro, name=name, **kwargs)

    with patch("backend.app.main.spawn_background_task", side_effect=spawn):
        return await _complete_concurrently(test_engine, printer_id, status, name=name)


@pytest.fixture(autouse=True)
def _clean():
    from backend.app.services.printer_manager import printer_manager

    main_module._print_plate_ids.clear()
    main_module._archive_runs_finishing.clear()
    yield
    main_module._print_plate_ids.clear()
    main_module._archive_runs_finishing.clear()
    printer_manager._awaiting_plate_clear.clear()


@pytest.fixture
async def farm(db_session, printer_factory):
    """A printer, a user paying from a cost centre, and an archive of a
    multi-plate file printed before."""
    db_session.add_all(
        [Settings(key="capture_finish_photo", value="false"), Settings(key="billing_enabled", value="true")]
    )
    printer = await printer_factory(name="P")
    user = User(username="payer", password_hash="x", is_active=True)
    center = CostCenter(name="Lab", is_active=True, is_private=False)
    db_session.add_all([user, center])
    await db_session.flush()
    archive = PrintArchive(
        printer_id=printer.id,
        filename=FILE,
        file_path=f"archives/test/{FILE}",
        file_size=1,
        print_name=SUBTASK,
        status="completed",
        cost=5.0,
        created_by_id=user.id,
    )
    db_session.add(archive)
    await db_session.commit()
    return SimpleNamespace(printer=printer.id, user=user.id, center=center.id, archive=archive.id)


async def _run(test_engine, db_session, farm, *, plate: int, status: str, run_id: str) -> int:
    """One queued print of the archive's plate, from dispatch to its end.
    Returns the queue item's id."""
    item = PrintQueueItem(
        printer_id=farm.printer,
        archive_id=farm.archive,
        status="printing",
        position=1,
        plate_id=plate,
        created_by_id=farm.user,
        cost_center_id=farm.center,
        estimated_cost=5.0,
        billing_run_id=run_id,
    )
    db_session.add(item)
    # As the scheduler's dispatch writes it (_start_print).
    (await db_session.get(PrintArchive, farm.archive)).plate_id = plate
    await db_session.commit()
    item_id = item.id
    register_expected_print(farm.printer, FILE, farm.archive, plate_id=plate, cost_center_id=farm.center)

    await _drive_print_start(test_engine, SimpleNamespace(id=farm.printer), download_ok=False)
    await _complete(test_engine, farm.printer, status, name=SUBTASK)
    return item_id


async def _runs(db_session, archive_id: int) -> list[PrintLogEntry]:
    db_session.expire_all()
    return list(
        (
            await db_session.scalars(
                select(PrintLogEntry).where(PrintLogEntry.archive_id == archive_id).order_by(PrintLogEntry.id)
            )
        ).all()
    )


async def test_a_queued_print_of_an_archive_is_one_run_on_it(test_engine, db_session, farm):
    item = await _run(test_engine, db_session, farm, plate=1, status="completed", run_id="run-1")

    runs = await _runs(db_session, farm.archive)
    assert [(r.status, r.plate_id, r.queue_item_id, r.printer_id) for r in runs] == [
        ("completed", 1, item, farm.printer)
    ]
    assert runs[0].started_at is not None and runs[0].completed_at is not None
    assert runs[0].completed_at >= runs[0].started_at
    archive = await db_session.get(PrintArchive, farm.archive)
    assert (archive.status, archive.plate_id) == ("completed", 1)
    assert await db_session.scalar(select(func.count()).select_from(PrintArchive)) == 1


async def test_each_run_keeps_its_own_plate_outcome_and_times(test_engine, db_session, farm):
    """Plate 1 fails, then plate 2 prints: neither run takes the other's."""
    first = await _run(test_engine, db_session, farm, plate=1, status="failed", run_id="run-1")
    second = await _run(test_engine, db_session, farm, plate=2, status="completed", run_id="run-2")

    runs = await _runs(db_session, farm.archive)
    assert [(r.status, r.plate_id, r.queue_item_id) for r in runs] == [
        ("failed", 1, first),
        ("completed", 2, second),
    ]
    assert runs[1].started_at >= runs[0].completed_at, "the second run's start is its own"
    assert runs[1].failure_reason is None, "a completed run carries no reason of the failed one"
    # The archive page shows the latest run.
    archive = await db_session.get(PrintArchive, farm.archive)
    assert (archive.status, archive.plate_id, archive.failure_reason) == ("completed", 2, None)


async def test_each_run_is_charged_once_under_its_own_run_id(test_engine, db_session, farm):
    await _run(test_engine, db_session, farm, plate=1, status="completed", run_id="run-1")
    await _run(test_engine, db_session, farm, plate=2, status="completed", run_id="run-2")

    charges = (
        await db_session.scalars(
            select(WalletTransaction)
            .where(WalletTransaction.transaction_type == "print_charge")
            .order_by(WalletTransaction.id)
        )
    ).all()
    assert [(c.print_run_id, c.print_archive_id, c.cost_center_id) for c in charges] == [
        ("run-1", farm.archive, farm.center),
        ("run-2", farm.archive, farm.center),
    ]


async def test_a_print_started_on_the_printer_is_an_archive_with_one_run(test_engine, db_session, printer_factory):
    db_session.add(Settings(key="capture_finish_photo", value="false"))
    await db_session.commit()
    printer_id = (await printer_factory(name="P")).id

    await _drive_print_start(test_engine, SimpleNamespace(id=printer_id), download_ok=False)
    archives = (await db_session.scalars(select(PrintArchive))).all()
    assert [r.status for r in await _all_runs(db_session, archives[0].id)] == ["printing"], "opened at its start"
    await _complete(test_engine, printer_id, "completed", name=SUBTASK)

    archives = (await db_session.scalars(select(PrintArchive))).all()
    assert len(archives) == 1
    runs = await _runs(db_session, archives[0].id)
    assert [(r.status, r.queue_item_id, r.printer_id) for r in runs] == [("completed", None, printer_id)]


async def test_a_reported_end_after_an_offline_stop_replaces_the_run(test_engine, db_session, farm, async_client):
    """The stop could not reach the printer and logged the run as cancelled;
    the printer then reports how it really ended. Still one run."""
    from unittest.mock import AsyncMock, patch

    item = PrintQueueItem(
        printer_id=farm.printer,
        archive_id=farm.archive,
        status="printing",
        position=1,
        plate_id=2,
        created_by_id=farm.user,
        billing_run_id="run-1",
    )
    db_session.add(item)
    await db_session.commit()
    register_expected_print(farm.printer, FILE, farm.archive, plate_id=2)
    await _drive_print_start(test_engine, SimpleNamespace(id=farm.printer), download_ok=False)

    with (
        patch("backend.app.services.print_control.stop_print_by_user", return_value=False),
        patch("backend.app.api.routes.print_queue.ws_manager") as ws,
    ):
        ws.send_archive_updated = AsyncMock()
        assert (await async_client.post(f"/api/v1/queue/{item.id}/stop")).status_code == 200
    assert [(r.status, r.plate_id) for r in await _runs(db_session, farm.archive)] == [("cancelled", 2)]

    main_module._active_prints[(farm.printer, FILE)] = farm.archive
    await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    assert [(r.status, r.plate_id) for r in await _runs(db_session, farm.archive)] == [("completed", 2)]


async def test_each_run_gets_its_own_energy(test_engine, db_session, farm):
    """The plug's counter at each start and end: the archive card shows the
    first run's, each run its own."""
    from unittest.mock import AsyncMock, MagicMock, patch

    plug = MagicMock(id=1)
    plug.name = "Plug"
    counters = iter([10.5, 20.3])

    async def reading(candidates, read, db):
        return plug, {"total": next(counters)}

    async def started_at(kwh):
        archive = await db_session.get(PrintArchive, farm.archive)
        archive.energy_start_kwh = kwh
        await db_session.commit()

    with (
        patch("backend.app.main.energy_plug_candidates", AsyncMock(return_value=[plug])),
        patch("backend.app.main.select_energy_reading", side_effect=reading),
        patch("backend.app.services.energy_price.print_energy_cost", AsyncMock(return_value=0.1)),
        patch("backend.app.services.energy_price.current_price", AsyncMock(return_value=0.3)),
    ):
        for plate, kwh in ((1, 10.0), (2, 20.0)):
            item = PrintQueueItem(
                printer_id=farm.printer, archive_id=farm.archive, status="printing", position=1, plate_id=plate
            )
            db_session.add(item)
            await db_session.commit()
            register_expected_print(farm.printer, FILE, farm.archive, plate_id=plate)
            await _drive_print_start(test_engine, SimpleNamespace(id=farm.printer), download_ok=False)
            await started_at(kwh)
            await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    runs = await _runs(db_session, farm.archive)
    assert [(r.plate_id, r.energy_kwh) for r in runs] == [(1, 0.5), (2, pytest.approx(0.3))]
    archive = await db_session.get(PrintArchive, farm.archive)
    assert archive.energy_kwh == 0.5


async def test_each_run_keeps_its_own_verdict(test_engine, db_session, farm):
    """An answered outcome prompt belongs to its run; the next run asks again."""
    from backend.app.services.print_confirmation import apply_outcome_verdict

    archive = await db_session.get(PrintArchive, farm.archive)
    archive.confirm_requested = True
    await db_session.commit()

    await _run(test_engine, db_session, farm, plate=1, status="completed", run_id="run-1")
    archive = await db_session.get(PrintArchive, farm.archive)
    assert await apply_outcome_verdict(db_session, archive, "reject", source="dialog")
    await db_session.commit()
    await _run(test_engine, db_session, farm, plate=2, status="completed", run_id="run-2")

    runs = await _runs(db_session, farm.archive)
    assert [(r.plate_id, r.user_verdict) for r in runs] == [(1, "reject"), (2, None)]
    archive = await db_session.get(PrintArchive, farm.archive)
    assert archive.user_verdict is None, "the latest run is not judged yet"


async def _all_runs(db_session, archive_id: int) -> list[PrintLogEntry]:
    from backend.app.models.print_log import INCLUDE_OPEN_RUNS

    db_session.expire_all()
    return list(
        (
            await db_session.scalars(
                select(PrintLogEntry)
                .where(PrintLogEntry.archive_id == archive_id)
                .order_by(PrintLogEntry.id)
                .execution_options(**{INCLUDE_OPEN_RUNS: True})
            )
        ).all()
    )


async def _start_queued(test_engine, db_session, farm, *, plate: int, run_id: str) -> int:
    item = PrintQueueItem(
        printer_id=farm.printer,
        archive_id=farm.archive,
        status="printing",
        position=1,
        plate_id=plate,
        created_by_id=farm.user,
        cost_center_id=farm.center,
        billing_run_id=run_id,
    )
    db_session.add(item)
    (await db_session.get(PrintArchive, farm.archive)).plate_id = plate
    await db_session.commit()
    register_expected_print(farm.printer, FILE, farm.archive, plate_id=plate, cost_center_id=farm.center)
    await _drive_print_start(test_engine, SimpleNamespace(id=farm.printer), download_ok=False)
    return item.id


async def test_the_run_is_written_when_the_print_starts(test_engine, db_session, farm, async_client):
    """Open, it is in no statistic, log or count; its end fills in the same run."""
    item = await _start_queued(test_engine, db_session, farm, plate=2, run_id="run-1")

    [run] = await _all_runs(db_session, farm.archive)
    assert (run.status, run.plate_id, run.queue_item_id, run.billing_run_id, run.cost_center_id, run.printer_id) == (
        "printing",
        2,
        item,
        "run-1",
        farm.center,
        farm.printer,
    )
    archive = await db_session.get(PrintArchive, farm.archive)
    assert run.started_at == archive.started_at
    assert await _runs(db_session, farm.archive) == [], "an open run is not a print that happened"
    log = (await async_client.get("/api/v1/print-log/")).json()
    assert log["total"] == 0 and log["items"] == []
    stats = (await async_client.get("/api/v1/archives/stats")).json()
    assert stats["total_prints"] == 0

    await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    assert [(r.id, r.status, r.plate_id) for r in await _all_runs(db_session, farm.archive)] == [
        (run.id, "completed", 2)
    ]


async def test_a_print_start_seen_again_opens_no_second_run(test_engine, db_session, farm):
    """A restart resumes the running print: the run it opened stays the one."""
    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")
    archive = await db_session.get(PrintArchive, farm.archive)
    archive.subtask_id = "job-7"
    await db_session.commit()
    main_module._active_prints.clear()

    await _drive_print_start(
        test_engine,
        SimpleNamespace(id=farm.printer),
        download_ok=False,
        state_fields={"subtask_id": "job-7", "progress": 50.0},
    )

    assert await db_session.scalar(select(func.count()).select_from(PrintArchive)) == 1, "resumed, not archived anew"
    assert [r.status for r in await _all_runs(db_session, farm.archive)] == ["printing"]


async def test_the_runs_energy_start_is_its_own(test_engine, db_session, farm):
    from unittest.mock import AsyncMock, MagicMock, patch

    from backend.app.main import _record_energy_start

    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")
    plug = MagicMock(id=4)
    plug.name = "Plug"
    archive = await db_session.get(PrintArchive, farm.archive)
    with (
        patch("backend.app.main.energy_plug_candidates", AsyncMock(return_value=[plug])),
        patch("backend.app.main.select_energy_reading", AsyncMock(return_value=(plug, {"total": 12.5}))),
        patch("backend.app.services.energy_price.current_price", AsyncMock(return_value=0.3)),
    ):
        assert await _record_energy_start(archive, farm.printer, db_session)

    [run] = await _all_runs(db_session, farm.archive)
    assert (run.energy_start_kwh, run.energy_start_plug_id, run.energy_start_price) == (12.5, 4, 0.3)
    archive = await db_session.get(PrintArchive, farm.archive)
    assert run.energy_start_at == archive.energy_start_at


async def test_an_open_run_is_left_out_of_reads_updates_and_deletes(db_session, farm):
    from sqlalchemy import delete, update

    from backend.app.models.print_log import INCLUDE_OPEN_RUNS

    db_session.add_all(
        [
            PrintLogEntry(archive_id=farm.archive, status="completed"),
            PrintLogEntry(archive_id=farm.archive, status="printing"),
        ]
    )
    await db_session.commit()

    assert await db_session.scalar(select(func.count(PrintLogEntry.id))) == 1
    await db_session.execute(update(PrintLogEntry).values(notes="edited"))
    await db_session.execute(delete(PrintLogEntry))
    await db_session.commit()

    left = (
        await db_session.execute(
            select(PrintLogEntry.status, PrintLogEntry.notes).execution_options(**{INCLUDE_OPEN_RUNS: True})
        )
    ).all()
    assert left == [("printing", None)]


@pytest.mark.parametrize("by", ["subtask", "name"])
async def test_a_print_running_since_before_the_upgrade_gets_its_run(test_engine, db_session, farm, by):
    """Bambuddy restarts into the new version mid-print: the archive is
    printing and has no run; resuming the print opens it."""
    from datetime import datetime, timedelta

    archive = await db_session.get(PrintArchive, farm.archive)
    archive.status = "printing"
    started = datetime.utcnow().replace(microsecond=0) - timedelta(minutes=20)
    archive.started_at = started
    archive.subtask_id = "job-7"
    await db_session.commit()

    await _drive_print_start(
        test_engine,
        SimpleNamespace(id=farm.printer),
        download_ok=False,
        state_fields={"progress": 40.0},
        raw_data={"subtask_id": "job-7"} if by == "subtask" else None,
    )

    assert await db_session.scalar(select(func.count()).select_from(PrintArchive)) == 1, "resumed, not archived anew"
    runs = await _all_runs(db_session, farm.archive)
    assert [(r.status, r.subtask_id) for r in runs] == [("printing", "job-7")]
    assert runs[0].started_at == started


async def test_a_downloaded_print_opens_its_run(test_engine, db_session, printer_factory):
    """Started on the printer, its 3MF fetched: the auto-archive path."""
    printer_id = (await printer_factory(name="P")).id

    await _drive_print_start(test_engine, SimpleNamespace(id=printer_id), download_ok=True)

    archives = (await db_session.scalars(select(PrintArchive))).all()
    assert len(archives) == 1
    assert [r.status for r in await _all_runs(db_session, archives[0].id)] == ["printing"]


async def test_two_plates_on_two_printers_at_once_leave_no_open_run(test_engine, db_session, farm, printer_factory):
    """The second start moves the archive's start to its own; the first
    printer's end still fills its own run, with its own start.

    The second printer's end finds the archive no longer printing and logs
    nothing, as before runs were opened at the start: the archive holds one
    print's state. Its open run goes with that end. Moving the print's state
    onto the run (P1b) is what gives it its own record."""
    other = (await printer_factory(name="Q")).id
    register_expected_print(farm.printer, FILE, farm.archive, plate_id=1)
    await _drive_print_start(test_engine, SimpleNamespace(id=farm.printer), download_ok=False)
    [first] = await _all_runs(db_session, farm.archive)
    first_start = first.started_at
    await asyncio.sleep(1.1)  # the second start is a later second
    register_expected_print(other, FILE, farm.archive, plate_id=2)
    await _drive_print_start(test_engine, SimpleNamespace(id=other), download_ok=False)

    await _complete(test_engine, farm.printer, "completed", name=SUBTASK)
    await _complete(test_engine, other, "completed", name=SUBTASK)

    runs = await _all_runs(db_session, farm.archive)
    assert [(r.id, r.printer_id, r.status, r.started_at) for r in runs] == [
        (first.id, farm.printer, "completed", first_start)
    ]
    open_left = (
        await db_session.scalars(
            select(PrintLogEntry).where(PrintLogEntry.status == "printing").execution_options(include_open_runs=True)
        )
    ).all()
    assert open_left == []


async def test_a_run_is_dated_by_its_end(test_engine, db_session, farm):
    """The print log and statistics filter runs by when they were logged,
    which was always their end."""
    from datetime import datetime, timedelta

    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")
    [run] = await _all_runs(db_session, farm.archive)
    run.created_at = datetime(2026, 1, 1)  # opened long ago
    await db_session.commit()

    await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    [run] = await _all_runs(db_session, farm.archive)
    assert run.created_at > datetime.utcnow() - timedelta(minutes=5)


async def test_a_run_left_open_by_an_unseen_end_is_dropped_at_the_next_start(test_engine, db_session, farm):
    """A print whose end never reached its run (cancelled as stale at the next
    start): as before runs opened at the start, it leaves no run."""
    db_session.add(PrintLogEntry(archive_id=None, printer_id=farm.printer, status="printing"))
    await db_session.commit()

    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")

    runs = (
        await db_session.scalars(
            select(PrintLogEntry)
            .where(PrintLogEntry.printer_id == farm.printer)
            .execution_options(include_open_runs=True)
        )
    ).all()
    assert [(r.archive_id, r.status) for r in runs] == [(farm.archive, "printing")]


async def test_a_print_whose_archive_was_deleted_leaves_no_run(test_engine, db_session, farm):
    from sqlalchemy import update

    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")
    # Deleted while it printed (ON DELETE SET NULL takes the run's link).
    await db_session.execute(update(PrintLogEntry).values(archive_id=None).execution_options(include_open_runs=True))
    await db_session.delete(await db_session.get(PrintArchive, farm.archive))
    await db_session.commit()

    await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    left = (await db_session.scalars(select(PrintLogEntry).execution_options(include_open_runs=True))).all()
    assert left == []


async def test_deleting_an_archive_with_its_stats_takes_its_open_run(db_session, farm, async_client):
    db_session.add(PrintLogEntry(archive_id=farm.archive, printer_id=farm.printer, status="printing"))
    await db_session.commit()

    response = await async_client.delete(f"/api/v1/archives/{farm.archive}?purge_stats=true")
    assert response.status_code in (200, 204), response.text

    left = (await db_session.scalars(select(PrintLogEntry).execution_options(include_open_runs=True))).all()
    assert left == []


async def test_the_energy_goes_to_the_run_that_ended(test_engine, db_session, farm, printer_factory):
    """Its run, not the archive's newest: a run opened earlier ends after a
    later one of the archive was written."""
    from datetime import datetime
    from unittest.mock import AsyncMock, MagicMock, patch

    await _start_queued(test_engine, db_session, farm, plate=1, run_id="run-1")
    other = (await printer_factory(name="Q")).id
    later = PrintLogEntry(
        archive_id=farm.archive, printer_id=other, status="completed", started_at=datetime(2026, 10, 30, 9)
    )
    db_session.add(later)
    archive = await db_session.get(PrintArchive, farm.archive)
    archive.energy_start_kwh = 10.0
    await db_session.commit()
    plug = MagicMock(id=1)
    plug.name = "Plug"

    with (
        patch("backend.app.main.energy_plug_candidates", AsyncMock(return_value=[plug])),
        patch("backend.app.main.select_energy_reading", AsyncMock(return_value=(plug, {"total": 10.5}))),
        patch("backend.app.services.energy_price.print_energy_cost", AsyncMock(return_value=0.1)),
        patch("backend.app.services.energy_price.current_price", AsyncMock(return_value=0.3)),
    ):
        await _complete(test_engine, farm.printer, "completed", name=SUBTASK)

    runs = {r.printer_id: r.energy_kwh for r in await _all_runs(db_session, farm.archive)}
    assert runs == {farm.printer: 0.5, other: None}


async def test_a_reprint_after_an_unseen_end_opens_a_run_of_its_own(test_engine, db_session, farm):
    """The archive's earlier run on this printer is still open (its end never
    came); the new start is a new run, not that one carried on."""
    from datetime import datetime

    stale = PrintLogEntry(
        archive_id=farm.archive, printer_id=farm.printer, status="printing", started_at=datetime(2026, 10, 1, 8)
    )
    db_session.add(stale)
    await db_session.commit()

    await _start_queued(test_engine, db_session, farm, plate=2, run_id="run-2")

    [run] = await _all_runs(db_session, farm.archive)
    archive = await db_session.get(PrintArchive, farm.archive)
    assert run.started_at != datetime(2026, 10, 1, 8)
    assert (run.started_at, run.plate_id) == (archive.started_at, 2)
