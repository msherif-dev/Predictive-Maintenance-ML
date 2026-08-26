from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib


@dataclass
class ModelArtifact:
    """كائن يحفظ الموديل مع كافة الإعدادات والكتالوج الخاص به."""
    model: Any
    feature_columns: List[str]
    constant_columns: List[str]
    original_columns: List[str]
    rolling_window: int
    metadata: Dict[str, Any]


def save_artifact(
    artifact: ModelArtifact, path: Union[str, Path]
) -> None:
    """حفظ الـ Artifact على القرص."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Union[str, Path]) -> ModelArtifact:
    """تحميل الـ Artifact وقراءته."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, ModelArtifact):
        raise TypeError(
            f"Expected ModelArtifact, got {type(artifact).__name__}"
        )

    return artifact