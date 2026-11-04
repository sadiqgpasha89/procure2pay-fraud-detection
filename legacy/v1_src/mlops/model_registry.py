"""Model Registry managing artifact versioning, metadata, and deployment lifecycle."""
import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ModelStage(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


class ModelMetadata(BaseModel):
    """Metadata detailing model lineage and offline benchmark scores."""
    model_name: str
    version: str
    algorithm: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    auc_pr: float = Field(..., ge=0.0, le=1.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)
    training_sample_count: int
    created_at: str
    artifact_path: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    """Enterprise MLOps Model Registry."""

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = registry_dir
        self.models: Dict[str, ModelMetadata] = {}
        self._load_local_metadata()

    def _load_local_metadata(self):
        """Load registered models from disk metadata if exists."""
        meta_file = os.path.join(self.registry_dir, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.models[k] = ModelMetadata(**v)
            except Exception as e:
                logger.error("model_registry.load_failed", error=str(e))

    def _save_local_metadata(self):
        """Persist registry metadata to disk."""
        os.makedirs(self.registry_dir, exist_ok=True)
        meta_file = os.path.join(self.registry_dir, "metadata.json")
        try:
            with open(meta_file, "w") as f:
                json.dump({k: v.model_dump() for k, v in self.models.items()}, f, indent=2)
        except Exception as e:
            logger.error("model_registry.save_failed", error=str(e))

    def register_model(
        self,
        model_name: str,
        version: str,
        algorithm: str,
        auc_pr: float,
        brier_score: float,
        training_sample_count: int,
        artifact_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        stage: ModelStage = ModelStage.SHADOW,
    ) -> ModelMetadata:
        """Register a new candidate model version."""
        key = f"{model_name}:{version}"
        meta = ModelMetadata(
            model_name=model_name,
            version=version,
            algorithm=algorithm,
            stage=stage,
            auc_pr=auc_pr,
            brier_score=brier_score,
            training_sample_count=training_sample_count,
            created_at=datetime.utcnow().isoformat() + "Z",
            artifact_path=artifact_path,
            hyperparameters=hyperparameters or {},
        )
        self.models[key] = meta
        self._save_local_metadata()
        logger.info("model_registry.model_registered", key=key, stage=stage.value, auc_pr=auc_pr)
        return meta

    def promote_model(self, model_name: str, version: str, target_stage: ModelStage):
        """Transition model deployment stage (e.g. from SHADOW to PRODUCTION)."""
        key = f"{model_name}:{version}"
        if key in self.models:
            # If promoting to PRODUCTION, demote existing production model to ARCHIVED
            if target_stage == ModelStage.PRODUCTION:
                for k, m in self.models.items():
                    if m.model_name == model_name and m.stage == ModelStage.PRODUCTION:
                        m.stage = ModelStage.ARCHIVED

            self.models[key].stage = target_stage
            self._save_local_metadata()
            logger.info("model_registry.stage_promoted", key=key, new_stage=target_stage.value)

    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Fetch current active production model."""
        for m in self.models.values():
            if m.model_name == model_name and m.stage == ModelStage.PRODUCTION:
                return m
        return None
