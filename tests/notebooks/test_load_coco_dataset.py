from types import SimpleNamespace

import pytest

from notebooks import load_coco_dataset as coco


def test_load_split_filters_classes_except_for_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_zoo_dataset(name: str, **kwargs: object) -> object:
        captured["name"] = name
        captured.update(kwargs)
        return SimpleNamespace(count=lambda: 3, name="sample-dataset")

    monkeypatch.setattr(coco.foz, "load_zoo_dataset", fake_load_zoo_dataset)

    dataset = coco.load_split("validation", sample_size=5)

    assert dataset.name == "sample-dataset"
    assert captured["name"] == "coco-2017"
    assert captured["max_samples"] == 5
    assert captured["classes"] == coco.ADAS_TARGETS
    assert captured["only_matching"] is True


def test_load_split_omits_class_filter_for_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_zoo_dataset(name: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(count=lambda: 1, name="test-dataset")

    monkeypatch.setattr(coco.foz, "load_zoo_dataset", fake_load_zoo_dataset)

    coco.load_split("test", sample_size=2)

    assert "classes" not in captured
    assert "only_matching" not in captured


@pytest.mark.parametrize("image_count", [0, 4])
def test_load_split_rejects_unexpected_image_count(
    image_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coco.foz,
        "load_zoo_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            count=lambda: image_count,
            name="unexpected-dataset",
        ),
    )

    with pytest.raises(ValueError, match="COCO split 'train'"):
        coco.load_split("train", sample_size=3)


def test_load_split_adds_context_to_network_and_io_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(*args: object, **kwargs: object) -> None:
        raise OSError("storage unavailable")

    monkeypatch.setattr(coco.foz, "load_zoo_dataset", fail_to_load)

    with pytest.raises(RuntimeError, match="Unable to load COCO split 'train'") as error:
        coco.load_split("train", sample_size=3)

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("split", "sample_size", "message"),
    [
        ("unknown", 1, "Unsupported COCO split"),
        ("train", 0, "sample size must be positive"),
    ],
)
def test_load_split_validates_arguments(
    split: str,
    sample_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coco.load_split(split, sample_size)
