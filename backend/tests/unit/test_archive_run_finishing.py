"""An archive stays marked while its last run's timelapse work goes on.

The timelapse scan and the layer timelapse stitch run after the print
completed and tell their run from a later one by the archive's start time. A
queue dispatch of the same file adds its run to the archive only once that
work is over (``_claim_reusable_archive``), or the work would stop short and
the video never be attached.
"""

import asyncio
from unittest.mock import patch

import pytest

from backend.app import main as main_module
from backend.tests.unit.test_run_estimate_plate import _complete


@pytest.fixture(autouse=True)
def _clean_state():
    main_module._archive_runs_finishing.clear()
    main_module._active_prints.clear()
    main_module._print_plate_ids.clear()
    yield
    main_module._archive_runs_finishing.clear()
    main_module._active_prints.clear()
    main_module._print_plate_ids.clear()


@pytest.mark.asyncio
async def test_marked_from_the_spawn_until_the_work_ends():
    release = asyncio.Event()

    async def work():
        await release.wait()

    wrapped = main_module._mark_run_finishing(7, work())
    assert main_module._archive_runs_finishing == {7: 1}
    second = main_module._mark_run_finishing(7, asyncio.sleep(0))
    await second
    assert main_module._archive_runs_finishing == {7: 1}, "the other task is still running"

    task = asyncio.create_task(wrapped)
    release.set()
    await task
    assert main_module._archive_runs_finishing == {}


@pytest.mark.asyncio
async def test_a_failing_task_unmarks_too():
    async def work():
        raise RuntimeError("scan failed")

    with pytest.raises(RuntimeError):
        await main_module._mark_run_finishing(7, work())
    assert main_module._archive_runs_finishing == {}


def test_the_timelapse_scan_marks_its_archive():
    spawned = []
    with patch("backend.app.main.spawn_background_task", side_effect=lambda coro, name: spawned.append(coro)):
        main_module._schedule_timelapse_scan(12, 3, plate_id=None)
    assert main_module._archive_runs_finishing == {12: 1}
    for coro in spawned:
        coro.close()


@pytest.mark.asyncio
async def test_the_layer_timelapse_of_a_completed_print_marks_its_archive(
    test_engine, db_session, printer_factory, archive_factory
):
    printer = await printer_factory()
    archive = await archive_factory(printer.id, filename="multi.gcode.3mf", status="printing", with_run=False)
    main_module._active_prints[(printer.id, "multi.gcode.3mf")] = archive.id
    spawned: list[str] = []
    real_spawn = main_module.spawn_background_task

    def spawn(coro, name=None, **kwargs):
        if name == "background-layer-timelapse":
            spawned.append(main_module._archive_runs_finishing.get(archive.id, 0))
        return real_spawn(coro, name=name, **kwargs)

    with patch("backend.app.main.spawn_background_task", side_effect=spawn):
        await _complete(test_engine, printer.id)

    assert spawned == [1]
