"""Service for writing independent print log entries.

Log entries are written to a separate table and never touch archives or queue items.
"""

import logging
from collections.abc import Set as AbstractSet
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.print_log import INCLUDE_OPEN_RUNS, RUN_OPEN, PrintLogEntry
from backend.app.utils.local_time import utcnow_naive

logger = logging.getLogger(__name__)


def wear_cost_for_run(duration_seconds: int | None, wear_cost_per_hour: float | None) -> float | None:
    """Printer wear for one run: its duration at the printer's hourly rate (#694).

    None when the printer has no rate or the run has no measured duration (a
    reconciled run stores 0 because its real end time is unknown).
    """
    if not isinstance(wear_cost_per_hour, int | float) or not isinstance(duration_seconds, int):
        return None
    if wear_cost_per_hour <= 0 or duration_seconds <= 0:
        return None
    return round(duration_seconds / 3600 * wear_cost_per_hour, 3)


async def write_log_entry(
    db: AsyncSession,
    *,
    status: str,
    archive_id: int | None = None,
    queue_item_id: int | None = None,
    print_name: str | None = None,
    plate_id: int | None = None,
    printer_name: str | None = None,
    printer_id: int | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    filament_type: str | None = None,
    filament_color: str | None = None,
    filament_used_grams: float | None = None,
    cost: float | None = None,
    energy_kwh: float | None = None,
    energy_cost: float | None = None,
    wear_cost_per_hour: float | None = None,
    failure_reason: str | None = None,
    thumbnail_path: str | None = None,
    created_by_id: int | None = None,
    created_by_username: str | None = None,
    reconciled: bool = False,
    duration_unmeasured: bool = False,
) -> PrintLogEntry:
    """Write a print log entry.

    ``reconciled`` marks a synthetic completion written when a stale
    ``status="printing"`` archive is closed out at reconnect. Its real end time
    is unknown — the print stopped somewhere during the disconnect and
    ``completed_at`` is only the reconnect moment — so ``completed_at -
    started_at`` would bank the entire disconnect gap as print time, adding
    hundreds of fictitious hours across a farm of stale rows (#2592). For those
    entries we store an explicit ``0`` ("no measured runtime") rather than a
    fabricated duration; the stats total trusts a stored 0 instead of
    recomputing from the stale timestamps.

    ``duration_unmeasured`` marks a run closed at reconnect, whose times were
    not measured (#3310): ``completed_at`` is its start plus the slicer's
    estimate, or the reconnect. The time accuracy leaves it out.
    """
    entry = await _entry_of_this_run(db, archive_id, printer_id, started_at)
    filling_open_run = entry is not None and entry.status == RUN_OPEN
    if entry is not None and filling_open_run and entry.started_at is not None:
        # The run's own start. The archive's is its latest run's: another
        # plate of it may have started on another printer since.
        started_at = entry.started_at

    if reconciled:
        duration: int | None = 0
    elif started_at and completed_at:
        duration = int((_utc(completed_at) - _utc(started_at)).total_seconds())
    else:
        duration = None

    fields = {
        "archive_id": archive_id,
        "queue_item_id": queue_item_id,
        "print_name": print_name,
        "plate_id": plate_id,
        "printer_name": printer_name,
        "printer_id": printer_id,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration,
        "filament_type": filament_type,
        "filament_color": filament_color,
        "filament_used_grams": filament_used_grams,
        "cost": cost,
        "energy_kwh": energy_kwh,
        "energy_cost": energy_cost,
        "wear_cost": wear_cost_for_run(duration, wear_cost_per_hour),
        "duration_unmeasured": True if duration_unmeasured else None,
        "failure_reason": failure_reason,
        "thumbnail_path": thumbnail_path,
        "created_by_id": created_by_id,
        "created_by_username": created_by_username,
    }
    if filling_open_run:
        # When the run was logged, as a run written at its end always was:
        # the date ranges of statistics and the print log go by it.
        fields["created_at"] = utcnow_naive()
    if entry is None:
        if archive_id is not None and started_at is not None and status != RUN_OPEN:
            # Shadow check while the run's start moves over (archive redesign):
            # every print start that binds an archive opens its run, so a run
            # ending without one is a start path that does not yet.
            logger.info("[RUN] Archive %s: no run was opened when the print started; written at its end", archive_id)
        entry = PrintLogEntry(**fields)
        db.add(entry)
    elif entry.status != status:
        if entry.status == RUN_OPEN and None not in (entry.plate_id, plate_id) and entry.plate_id != plate_id:
            logger.warning(
                "[RUN] Archive %s: the run opened for plate %s ends as plate %s", archive_id, entry.plate_id, plate_id
            )
        # The run ended differently from how it was first logged: a print
        # stopped while its printer was offline that finished after all. The
        # real outcome replaces the stop rather than counting the run twice.
        for name, value in fields.items():
            if value is None and name in _KEPT_WHEN_NOT_REPORTED:
                continue
            setattr(entry, name, value)
    # The same outcome reported again (a repeated completion after a reconnect)
    # changes nothing: the repeat no longer has the run's tracked usage, and the
    # entry may have been edited or given its energy since.
    await db.flush()
    return entry


async def mirror_archive_cost(
    db: AsyncSession, archive, previous_cost: float | None, *, skip_runs: AbstractSet[int] = frozenset()
) -> None:
    """Give the archive's newest run the archive's new cost, if it had the old one.

    Statistics, project totals and batch sums read a run's cost. A completed
    run without its own spool price took the archive's (or none), so a new
    archive cost belongs to it too. A run its spools priced keeps that price,
    and a failed or cancelled run keeps its own: it printed only part of it.
    A run in ``skip_runs`` was given its own new cost already.
    """
    entry = await db.scalar(
        select(PrintLogEntry).where(PrintLogEntry.archive_id == archive.id).order_by(PrintLogEntry.id.desc()).limit(1)
    )
    if entry is None or entry.id in skip_runs:
        return
    if entry.status == "completed" and entry.cost in (None, previous_cost):
        entry.cost = archive.cost


async def other_runs_of(db: AsyncSession, archive) -> int:
    """How many log entries *archive* has from runs other than its current one.

    The archive card keeps its first run's figures; a run whose own entry is
    already written (its stop, logged while the printer was offline) is still
    the first run.
    """
    # Without a start time or printer on the archive, ``!= None`` reads as
    # ``IS NOT NULL``: every entry is another run's.
    query = select(func.count(PrintLogEntry.id)).where(
        PrintLogEntry.archive_id == archive.id,
        or_(
            PrintLogEntry.started_at.is_(None),
            PrintLogEntry.started_at != archive.started_at,
            PrintLogEntry.printer_id.is_(None),
            PrintLogEntry.printer_id != archive.printer_id,
        ),
    )
    return await db.scalar(query) or 0


# Who and what the run was, which a repeated report may no longer know: a
# completion that finds the queue item already closed reports none. Its
# outcome (status, reason, grams, cost) is replaced as reported.
_KEPT_WHEN_NOT_REPORTED = frozenset(
    {
        "queue_item_id",
        "print_name",
        "plate_id",
        "printer_name",
        "thumbnail_path",
        "created_by_id",
        "created_by_username",
    }
)


def _utc(moment: datetime) -> datetime:
    """Naive UTC, as the columns store it; an aware time is converted first."""
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment


async def _entry_of_this_run(
    db: AsyncSession,
    archive_id: int | None,
    printer_id: int | None,
    started_at: datetime | None,
    *,
    exact: bool = False,
) -> PrintLogEntry | None:
    """The entry already written for this run, if any.

    A run is its archive, printer and start time together: plates of one
    archive printing on two printers share the archive row (and can share its
    start time), and a reprint starts at a new time. The queue item is not
    part of it: a repeated completion finds the item already closed and
    reports none. Without an archive or printer a run can't be told apart,
    so nothing is matched.

    A run opened at its start is also found as the archive's run open on the
    printer, unless ``exact``: the archive's start is its latest run's, and
    another plate of it may have started on another printer since.
    """
    if archive_id is None or printer_id is None:
        return None
    if started_at is None:
        return None if exact else await _open_run_on(db, archive_id, printer_id)
    query = (
        select(PrintLogEntry)
        .where(
            PrintLogEntry.archive_id == archive_id,
            PrintLogEntry.printer_id == printer_id,
            PrintLogEntry.started_at == started_at,
        )
        # The run opened when the print started is the one this fills in.
        .execution_options(**{INCLUDE_OPEN_RUNS: True})
    )
    entry = (await db.scalars(query.order_by(PrintLogEntry.id.desc()).limit(1))).first()
    # The archive's start is its latest run's. With another plate of it
    # started on another printer since, this printer's run is found as the
    # one open there.
    if entry is not None or exact:
        return entry
    return await _open_run_on(db, archive_id, printer_id)


async def _open_run_on(db: AsyncSession, archive_id: int, printer_id: int) -> PrintLogEntry | None:
    """The run of the archive open on the printer: there is one at most, a
    printer prints one thing at a time (``open_run``)."""
    return await db.scalar(
        select(PrintLogEntry)
        .where(
            PrintLogEntry.archive_id == archive_id,
            PrintLogEntry.printer_id == printer_id,
            PrintLogEntry.status == RUN_OPEN,
        )
        .order_by(PrintLogEntry.id.desc())
        .limit(1)
        .execution_options(**{INCLUDE_OPEN_RUNS: True})
    )


async def open_run_on_printer(
    db: AsyncSession, printer_id: int, *, subtask_id: str | None = None
) -> PrintLogEntry | None:
    """The run open on the printer: the print it is running now, which
    Bambuddy saw start. One at most (``open_run``). With ``subtask_id``, only
    a run of that job."""
    query = select(PrintLogEntry).where(PrintLogEntry.printer_id == printer_id, PrintLogEntry.status == RUN_OPEN)
    if subtask_id is not None:
        query = query.where(PrintLogEntry.subtask_id == subtask_id)
    return await db.scalar(
        query.order_by(PrintLogEntry.id.desc()).limit(1).execution_options(**{INCLUDE_OPEN_RUNS: True})
    )


async def discard_open_runs(db: AsyncSession, printer_id: int) -> None:
    """Drop the runs left open on the printer by prints that ended without
    their end reaching a run: cancelled at the next start as stale, or whose
    archive was deleted. As before runs were opened at the start, such a
    print leaves no run. Does not commit."""
    await db.execute(
        delete(PrintLogEntry)
        .where(PrintLogEntry.printer_id == printer_id, PrintLogEntry.status == RUN_OPEN)
        .execution_options(**{INCLUDE_OPEN_RUNS: True})
    )


async def open_run(
    db: AsyncSession,
    archive,
    *,
    printer_id: int,
    printer_name: str | None = None,
    plate_id: int | None = None,
) -> PrintLogEntry | None:
    """The run of the print that just started on ``archive``, written now with
    the open status; its end fills it in (``write_log_entry``).

    Idempotent: a print start seen again (a restart resuming the print, a
    repeated report) finds the run it opened, keyed as its end will key it, by
    archive, printer and start time. The queue item running it, if any, gives
    the dispatch's own billing run and cost centre. Does not commit.
    """
    from backend.app.models.print_queue import PrintQueueItem

    if archive.started_at is None:
        return None
    # Only the run of this very start: one still open on the printer from an
    # earlier start of the archive is of a print that ended unseen.
    entry = await _entry_of_this_run(db, archive.id, printer_id, archive.started_at, exact=True)
    if entry is not None:
        return entry
    # A printer prints one thing at a time: a run still open on it is of a
    # print that ended without its end reaching it.
    await discard_open_runs(db, printer_id)
    item = await db.scalar(
        select(PrintQueueItem)
        .where(
            PrintQueueItem.archive_id == archive.id,
            PrintQueueItem.printer_id == printer_id,
            PrintQueueItem.status == "printing",
        )
        .order_by(PrintQueueItem.id.desc())
        .limit(1)
    )
    entry = PrintLogEntry(
        status=RUN_OPEN,
        archive_id=archive.id,
        queue_item_id=item.id if item is not None else None,
        print_name=archive.print_name,
        plate_id=plate_id if plate_id is not None else archive.plate_id,
        printer_name=printer_name,
        printer_id=printer_id,
        started_at=archive.started_at,
        filament_type=archive.filament_type,
        filament_color=archive.filament_color,
        thumbnail_path=archive.thumbnail_path,
        created_by_id=archive.created_by_id,
        subtask_id=archive.subtask_id,
        billing_run_id=(item.billing_run_id if item is not None else None) or archive.billing_run_id,
        cost_center_id=(item.cost_center_id if item is not None else None) or archive.cost_center_id,
        confirm_requested=archive.confirm_requested,
        quantity=archive.quantity,
    )
    db.add(entry)
    await db.flush()
    return entry


async def mirror_energy_start(db: AsyncSession, archive, *, printer_id: int) -> None:
    """Copy the archive's energy start reading onto the run it was taken for.
    Does not commit."""
    entry = await _entry_of_this_run(db, archive.id, printer_id, archive.started_at)
    if entry is None or entry.status != RUN_OPEN:
        return
    entry.energy_start_kwh = archive.energy_start_kwh
    entry.energy_start_at = archive.energy_start_at
    entry.energy_start_plug_id = archive.energy_start_plug_id
    entry.energy_start_price = archive.energy_start_price


async def record_archive_wear(db: AsyncSession, archive, entry: PrintLogEntry) -> None:
    """Copy a run's wear cost onto its archive when it is the archive's first run.

    Same rule as energy (#1378): the archive shows its first print, and a
    reprint's wear stays on its own log entry (#694). ``entry`` must already be
    flushed, so it counts as one of the archive's runs.
    """
    if entry.wear_cost is None:
        return
    runs = await db.scalar(select(func.count(PrintLogEntry.id)).where(PrintLogEntry.archive_id == archive.id))
    if (runs or 0) <= 1:
        archive.wear_cost = entry.wear_cost
