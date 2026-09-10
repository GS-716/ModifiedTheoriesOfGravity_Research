from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "TensorEngine" / "src"))

from ResearchWorkflow.cleanup import clean_outputs


def test_clean_outputs_removes_only_expected_output_children(tmp_path):
    workflow = tmp_path / "ResearchWorkflow"
    output_root = workflow / "outputs"
    bundle = output_root / "notebook_cases" / "case-a"
    bundle.mkdir(parents=True)
    (bundle / "results.json").write_text("{}", encoding="utf-8")
    (output_root / "loose.log").write_text("diagnostic", encoding="utf-8")
    output_readme = output_root / "README.md"
    output_readme.write_text("preserve output documentation", encoding="utf-8")
    sentinel = workflow / "README.md"
    sentinel.write_text("preserve", encoding="utf-8")

    result = clean_outputs(output_root, expected_root=output_root)

    assert result.removed == ("loose.log", "notebook_cases")
    assert result.removed_count == 2
    assert output_root.is_dir()
    assert list(output_root.iterdir()) == [output_readme]
    assert output_readme.read_text(encoding="utf-8") == "preserve output documentation"
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_clean_outputs_rejects_any_root_other_than_the_expected_outputs(tmp_path):
    expected = tmp_path / "ResearchWorkflow" / "outputs"
    unexpected = tmp_path / "other" / "outputs"
    unexpected.mkdir(parents=True)
    marker = unexpected / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="Limpieza rechazada"):
        clean_outputs(unexpected, expected_root=expected)

    assert marker.read_text(encoding="utf-8") == "preserve"
