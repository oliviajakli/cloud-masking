from pathlib import Path

from src.bootstrap import list_scenes


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"data")


def test_list_scenes_intersection(tmp_path):
    gt = tmp_path / "gt"
    alg = tmp_path / "alg"
    gt.mkdir()
    alg.mkdir()

    touch(gt / "a.tif")
    touch(gt / "b.tif")
    touch(alg / "b.tif")
    touch(alg / "c.tif")

    scenes = list_scenes(gt, alg)
    assert scenes == ["b.tif"]
