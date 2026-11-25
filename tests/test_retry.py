"""Retry checkpoints, invalidation, and safe result publication."""

from pathlib import Path

import pytest
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.pipeline import STAGES, Stage
from autoclip.pipeline import export as export_module
from autoclip.pipeline.reframe.croppath import CropPath
from autoclip.pipeline.runner import JobWorkspace, PipelineRunner
from autoclip.pipeline.transcript import Transcript


def test_retry_checkpoint_round_trips_without_a_schema_change() -> None:
    workspace = JobWorkspace("retry-checkpoint")

    workspace.set_retry_stage(Stage.REFRAME)

    assert workspace.get_retry_stage() is Stage.REFRAME
    workspace.clear_retry_stage()
    assert workspace.get_retry_stage() is None


def test_retry_regenerates_failed_stage_and_every_downstream_stage() -> None:
    job = Job(id=new_id(), source_id="source")
    source = Source(id="source", type="upload", path="video.mp4")
    workspace = JobWorkspace(job.id)
    workspace.set_retry_stage(Stage.HIGHLIGHTS)

    runner = PipelineRunner(job, source, settings=Settings())

    assert [stage for stage in STAGES if runner._regenerate(stage)] == [
        Stage.HIGHLIGHTS,
        Stage.REFRAME,
        Stage.CAPTIONS,
        Stage.EXPORT,
    ]


def test_failed_reexport_keeps_the_previous_complete_output(
    initialised_db: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"source")
    source = store.create_source(
        Source(id=new_id(), type="upload", path=str(source_file), title="source")
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id))
    clips = [
        Clip(id=new_id(), job_id=job.id, rank=rank, start_s=rank, end_s=rank + 1) for rank in (1, 2)
    ]
    store.replace_clips(job.id, clips)

    old_paths: list[Path] = []
    for clip in clips:
        old_path = tmp_path / f"old-{clip.rank}.mp4"
        old_path.write_bytes(f"old-{clip.rank}".encode())
        old_paths.append(old_path)
        store.create_export(Export(id=new_id(), clip_id=clip.id, path=str(old_path)))

    calls = 0

    def fail_after_one_render(request, **_kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise export_module.ExportError("encoder stopped")
        request.destination.write_bytes(b"replacement")

    monkeypatch.setattr(export_module, "export_clip", fail_after_one_render)
    runner = PipelineRunner(job, source, settings=Settings())
    crops = {clip.id: CropPath(source_width=1, source_height=1) for clip in clips}

    with pytest.raises(export_module.ExportError, match="encoder stopped"):
        runner._stage_export(clips, Transcript(), crops)

    assert [store.list_exports(clip.id)[0].path for clip in clips] == [
        str(path) for path in old_paths
    ]
    assert [path.read_bytes() for path in old_paths] == [b"old-1", b"old-2"]
