"""src/models 的資料結構與 src/evaluation 的評估流程。"""

import pytest

from src.evaluation import edit_distance, evaluate
from src.models import BatchOutput, BoundingBox, EvalReport, PlateResult, PredictionOutput

from conftest import write_jpg


def _plate(text, conf):
    return PlateResult(plate_text=text, confidence=conf, detector_score=conf, recognizer_score=1.0, bbox=[0, 0, 1, 1])


def test_bbox_and_best_plate():
    assert BoundingBox(x1=1, y1=2, x2=3, y2=4).to_list() == [1, 2, 3, 4]
    out = PredictionOutput(source_file="a.jpg", plates=[_plate("A", 0.3), _plate("B", 0.8)])
    assert out.best_plate().plate_text == "B"
    assert PredictionOutput(source_file="a.jpg").best_plate() is None
    assert len(out.job_id) == 36 and out.timestamp.endswith("+00:00")
    batch = BatchOutput(total_images=1, results=[out])
    assert BatchOutput.model_validate_json(batch.model_dump_json()) == batch


@pytest.mark.parametrize("a,b,d", [("", "", 0), ("ABC", "", 3), ("", "AB", 2), ("ABC-1234", "ABC-1234", 0),
                                   ("ABC-1234", "A8C-1234", 1), ("ABC", "ABCD", 1), ("kitten", "sitting", 3)])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == d


class _Pipeline:
    """依檔名回傳預先設定的車牌。"""

    def __init__(self, answers):
        self.answers = answers

    def predict(self, path):
        text = self.answers[path.name]
        plates = [] if text is None else [_plate(text, 0.9)]
        return PredictionOutput(source_file=path.name, plates=plates)


def test_evaluate_report(tmp_path):
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_jpg(tmp_path / name)
    gt = tmp_path / "gt.csv"
    gt.write_text("image_path,plate_text\na.jpg,ABC-1234\nb.jpg,XYZ-0001\nc.jpg,QQQ-1111\n", encoding="utf-8")
    pipeline = _Pipeline({"a.jpg": "ABC-1234", "b.jpg": "XYZ-0007", "c.jpg": None})

    report = evaluate(pipeline, gt)
    assert (report.total, report.correct, report.no_plate) == (3, 1, 1)
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.char_accuracy == pytest.approx(1 - (0 + 1 + 8) / 24)
    assert [s.source_file for s in report.mismatches()] == ["b.jpg", "c.jpg"]


def test_evaluate_empty_csv(tmp_path):
    gt = tmp_path / "gt.csv"
    gt.write_text("image_path,plate_text\n", encoding="utf-8")
    report = evaluate(_Pipeline({}), gt)
    assert (report.total, report.accuracy, report.char_accuracy) == (0, 0.0, 0.0)
    assert isinstance(report, EvalReport)
