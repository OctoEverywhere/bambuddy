"""A library file printed again adds its run to the file's archive.

Every queue dispatch of a library or project file used to archive the file
anew, so a two-plate file printed plate by plate showed two archives of the
same file, flagged as duplicates of each other. The dispatch now adds its run
to the archive the file already has: same file, same owner, same project, from
this library file (or one since deleted).

An archive holds one active run, and uploads run in parallel, so it is only
taken while idle: not printing, not sent or waiting to start, not queued, not
claimed by another dispatch still uploading. A print of the file running at
the same time gets an archive of its own, as before.
"""

import hashlib
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models  # noqa: F401 - populate Base.metadata
import backend.app.services.print_scheduler as scheduler_module
from backend.app import main as main_module
from backend.app.core.database import Base
from backend.app.models.archive import PrintArchive
from backend.app.models.library import LibraryFile
from backend.app.models.print_queue import PrintQueueItem
from backend.app.models.printer import Printer
from backend.app.services.print_scheduler import PrintScheduler
from backend.tests._fixtures.background_tasks import discarding_spawn_patch

CONTENT = b"two plates"
HASH = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _clean_print_state():
    main_module._active_prints.clear()
    main_module._expected_prints.clear()
    main_module._archive_runs_finishing.clear()
    yield
    main_module._active_prints.clear()
    main_module._expected_prints.clear()
    main_module._archive_runs_finishing.clear()


@pytest.fixture
async def case(tmp_path):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    source = tmp_path / "library" / "planter.3mf"
    source.parent.mkdir()
    source.write_bytes(CONTENT)
    async with session_maker() as db:
        printer = Printer(name="X1C", serial_number="SERIAL-1", ip_address="127.0.0.1", access_code="code", model="X1C")
        library_file = LibraryFile(
            filename="planter.3mf",
            file_path=str(source),
            file_type="3mf",
            file_size=source.stat().st_size,
            file_hash=HASH,
        )
        db.add_all([printer, library_file])
        await db.commit()
        ctx = SimpleNamespace(
            session_maker=session_maker,
            base_dir=tmp_path,
            printer_id=printer.id,
            library_file_id=library_file.id,
        )

    async def archive(**overrides) -> int:
        """An archive of the file, its copy on disk unless ``on_disk=False``."""
        on_disk = overrides.pop("on_disk", True)
        rel = Path("archive") / f"copy-{overrides.get('plate_id', 1)}-{len(overrides)}.3mf"
        async with session_maker() as db:
            fields = {
                "printer_id": ctx.printer_id,
                "filename": "planter.3mf",
                "file_path": str(rel),
                "file_size": 10,
                "content_hash": HASH,
                "status": "completed",
                "library_file_id": ctx.library_file_id,
                "plate_id": 1,
            }
            row = PrintArchive(**(fields | overrides))
            db.add(row)
            await db.commit()
            if on_disk:
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(CONTENT)
            return row.id

    async def item(**overrides) -> int:
        async with session_maker() as db:
            fields = {
                "printer_id": ctx.printer_id,
                "library_file_id": ctx.library_file_id,
                "status": "pending",
                "plate_id": 2,
            }
            row = PrintQueueItem(**(fields | overrides))
            db.add(row)
            await db.commit()
            return row.id

    ctx.archive = archive
    ctx.item = item
    with patch.object(scheduler_module.settings, "base_dir", tmp_path):
        yield ctx
    await engine.dispose()


async def _claim(ctx, scheduler, item_id):
    async with ctx.session_maker() as db:
        item = await db.get(PrintQueueItem, item_id)
        library_file = await db.get(LibraryFile, ctx.library_file_id)
        archive = await scheduler._claim_reusable_archive(db, item, Path(library_file.file_path))
        return archive.id if archive is not None else None


@pytest.mark.asyncio
async def test_the_files_archive_is_reused(case):
    existing = await case.archive()
    assert await _claim(case, PrintScheduler(), await case.item()) == existing


@pytest.mark.asyncio
async def test_the_file_is_matched_as_it_is_on_disk(case):
    """The stored hash is not refreshed when the file changes in place (an
    external folder): a stale one neither finds the old content's archive
    nor misses the current one."""
    existing = await case.archive()
    async with case.session_maker() as db:
        (await db.get(LibraryFile, case.library_file_id)).file_hash = "b" * 64
        await db.commit()
    assert await _claim(case, PrintScheduler(), await case.item()) == existing


@pytest.mark.asyncio
async def test_an_archive_whose_library_file_was_deleted_is_reused(case):
    existing = await case.archive(library_file_id=None)
    assert await _claim(case, PrintScheduler(), await case.item()) == existing


@pytest.mark.asyncio
async def test_the_newest_idle_archive_is_taken(case):
    await case.archive()
    newest = await case.archive(tags="newest")
    assert await _claim(case, PrintScheduler(), await case.item()) == newest


@pytest.mark.parametrize(
    "archive_overrides, item_overrides",
    [
        pytest.param({"content_hash": "b" * 64}, {}, id="another-file"),
        pytest.param({"created_by_id": 7}, {}, id="another-owner"),
        pytest.param({}, {"created_by_id": 7}, id="owned-print-of-an-unowned-archive"),
        pytest.param({"project_id": 3}, {}, id="another-project"),
        pytest.param({}, {"project_id": 3}, id="project-print-of-a-loose-archive"),
        pytest.param({"library_file_id": 999}, {}, id="another-library-file"),
        pytest.param({"deleted_at": datetime(2026, 10, 1)}, {}, id="deleted"),
        pytest.param({"status": "printing"}, {}, id="printing"),
        pytest.param({"on_disk": False}, {}, id="its-copy-is-gone"),
    ],
)
@pytest.mark.asyncio
async def test_no_archive_that_is_not_this_files_or_not_usable(case, archive_overrides, item_overrides):
    await case.archive(**archive_overrides)
    assert await _claim(case, PrintScheduler(), await case.item(**item_overrides)) is None


@pytest.mark.asyncio
async def test_not_while_another_queue_item_prints_or_waits_on_it(case):
    existing = await case.archive()
    await case.item(archive_id=existing, library_file_id=None, status="pending")
    assert await _claim(case, PrintScheduler(), await case.item()) is None


@pytest.mark.asyncio
async def test_not_while_one_of_its_prints_runs_or_finishes(case):
    existing = await case.archive()
    main_module._active_prints[(case.printer_id, "planter.3mf")] = existing
    assert await _claim(case, PrintScheduler(), await case.item()) is None

    # Completed, but its timelapse is still being looked for: that scan tells
    # its run by the archive's start time, which a new run would change.
    main_module._active_prints.clear()
    main_module._archive_runs_finishing[existing] = 1
    assert await _claim(case, PrintScheduler(), await case.item()) is None


@pytest.mark.asyncio
async def test_a_name_left_from_an_earlier_dispatch_does_not_block_it(case):
    """A print start leaves the '<name>.gcode' variant of its registration
    behind for two hours; the plate printed next must still join."""
    existing = await case.archive()
    main_module._expected_prints[(case.printer_id, "planter.gcode")] = existing
    assert await _claim(case, PrintScheduler(), await case.item()) == existing


@pytest.mark.asyncio
async def test_two_dispatches_in_flight_never_share_it(case):
    existing = await case.archive()
    scheduler = PrintScheduler()
    first, second = await case.item(), await case.item()

    assert await _claim(case, scheduler, first) == existing
    assert await _claim(case, scheduler, second) is None

    # The first dispatch ended: the archive is free for the next print.
    scheduler._release_archive_claims(first)
    assert await _claim(case, scheduler, second) == existing


async def _start(case, item_id, archive_print):
    scheduler = PrintScheduler()
    patches = [
        patch("backend.app.services.archive.ArchiveService.archive_print", new=archive_print),
        patch("backend.app.services.print_scheduler.printer_manager.is_connected", MagicMock(return_value=True)),
        patch("backend.app.services.print_scheduler.printer_manager.get_status", MagicMock(return_value=None)),
        patch("backend.app.services.print_scheduler.printer_manager.start_print", MagicMock(return_value=True)),
        patch("backend.app.services.print_scheduler.printer_manager.set_awaiting_plate_clear", MagicMock()),
        patch(
            "backend.app.services.print_scheduler.get_ftp_retry_settings", AsyncMock(return_value=(False, 0, 0, 1.0))
        ),
        patch("backend.app.services.print_scheduler.delete_file_async", AsyncMock(return_value=True)),
        patch("backend.app.services.print_scheduler.upload_file_async", AsyncMock(return_value=True)),
        patch("backend.app.services.print_scheduler.cache_3mf_download", MagicMock()),
        discarding_spawn_patch(),
        patch("backend.app.services.notification_service.notification_service.on_queue_job_started", AsyncMock()),
        patch("backend.app.services.notification_service.notification_service.on_queue_job_failed", AsyncMock()),
        patch("backend.app.services.mqtt_relay.mqtt_relay.on_queue_job_started", AsyncMock()),
        patch.object(scheduler, "_propagate_owner_to_printer_manager", AsyncMock()),
        patch.object(scheduler, "_power_off_if_needed", AsyncMock()),
    ]
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        async with case.session_maker() as db:
            item = await db.get(PrintQueueItem, item_id)
            await scheduler._start_print(db, item)
    async with case.session_maker() as db:
        item = await db.get(PrintQueueItem, item_id)
        return item, (await db.get(PrintArchive, item.archive_id) if item.archive_id else None)


@pytest.mark.asyncio
async def test_the_dispatch_adds_its_run_to_the_files_archive(case):
    existing = await case.archive(cost_center_id=5)
    archive_print = AsyncMock()

    item, archive = await _start(case, await case.item(plate_id=2), archive_print)

    archive_print.assert_not_called()
    assert item.archive_id == existing
    # The latest run's, as a new archive would have them.
    assert archive.plate_id == 2
    assert archive.cost_center_id is None


@pytest.mark.asyncio
async def test_a_file_never_archived_is_archived(case):
    created = {}

    async def archive_print(self, **kwargs):
        row = PrintArchive(
            printer_id=kwargs["printer_id"],
            filename="planter.3mf",
            file_path="archive/new.3mf",
            file_size=10,
            content_hash=HASH,
            status="archived",
        )
        self.db.add(row)
        await self.db.flush()
        created["id"] = row.id
        return row

    item, _archive = await _start(case, await case.item(), archive_print)

    assert item.archive_id == created["id"]


@pytest.mark.asyncio
async def test_a_reprint_of_another_plate_shows_that_plate(case):
    """The archive shows its latest run's plate; the first one's stays on its
    print log entry."""
    existing = await case.archive(plate_id=1)

    _item, archive = await _start(
        case, await case.item(archive_id=existing, library_file_id=None, plate_id=3), AsyncMock()
    )

    assert archive.plate_id == 3


@pytest.mark.parametrize("fails", [False, True], ids=["sent", "failed"])
@pytest.mark.asyncio
async def test_the_dispatch_gives_its_claim_back_when_it_ends(fails):
    """Sent, its print shows as running; failed, the archive is free again.
    Another dispatch's claim is left alone."""
    scheduler = PrintScheduler()

    async def start_print(db, item):
        scheduler._archive_claims[298] = 597
        if fails:
            raise RuntimeError("upload failed")

    scheduler._archive_claims[300] = 600
    db = MagicMock()
    db.get = AsyncMock(return_value=MagicMock(id=597))
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=db)
    session.__aexit__ = AsyncMock(return_value=False)
    with ExitStack() as stack:
        stack.enter_context(patch("backend.app.services.print_scheduler.async_session", return_value=session))
        stack.enter_context(patch.object(scheduler, "_claim_for_dispatch", AsyncMock(return_value=True)))
        stack.enter_context(patch.object(scheduler, "_start_print", side_effect=start_print))
        stack.enter_context(patch.object(scheduler, "_clear_dispatch_claim", AsyncMock()))
        if fails:
            stack.enter_context(pytest.raises(RuntimeError))
        await scheduler._dispatch_one(597)

    assert scheduler._archive_claims == {300: 600}
