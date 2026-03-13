"""PostgreSQL + SQLAlchemy 2.0 data model for an NLP sentiment-analysis ML platform.

This module covers key ML lifecycle entities:
- clients and users
- datasets and dataset versions
- model registry and model evaluation metrics
- inference logs for observability and drift monitoring
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload


class Base(DeclarativeBase):
    """Base class for all declarative SQLAlchemy models."""


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"


class InferenceEnvironment(str, enum.Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    EXPERIMENT = "experiment"


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str | None] = mapped_column(String(120))
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    users: Mapped[list[ClientUser]] = relationship(
        back_populates="client", cascade="all, delete-orphan", lazy="selectin"
    )
    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="client", cascade="all, delete-orphan", lazy="selectin"
    )
    registered_models: Mapped[list[RegisteredModel]] = relationship(
        back_populates="client", cascade="all, delete-orphan", lazy="selectin"
    )


class ClientUser(Base):
    __tablename__ = "client_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="viewer")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    client: Mapped[Client] = relationship(back_populates="users", lazy="joined")

    __table_args__ = (
        UniqueConstraint("client_id", "email", name="uq_client_users_client_email"),
        Index("ix_client_users_client_id", "client_id"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(180), nullable=False)
    task_type: Mapped[str] = mapped_column(
        String(80), nullable=False, default="nlp_sentiment_analysis"
    )
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    client: Mapped[Client] = relationship(back_populates="datasets", lazy="joined")
    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("client_id", "name", name="uq_datasets_client_name"),
        Index("ix_datasets_client_id", "client_id"),
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    labels: Mapped[list[str] | None] = mapped_column(ARRAY(String(64)))
    preprocessing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status"),
        nullable=False,
        default=ProcessingStatus.PENDING,
    )
    data_schema: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="versions", lazy="joined")
    model_versions: Mapped[list[ModelVersion]] = relationship(
        back_populates="dataset_version", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_versions_dataset_version"),
        Index("ix_dataset_versions_dataset_id", "dataset_id"),
        Index("ix_dataset_versions_status", "preprocessing_status"),
    )


class RegisteredModel(Base):
    __tablename__ = "registered_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    problem_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    client: Mapped[Client] = relationship(back_populates="registered_models", lazy="joined")
    versions: Mapped[list[ModelVersion]] = relationship(
        back_populates="registered_model", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("client_id", "model_name", name="uq_registered_models_client_name"),
        Index("ix_registered_models_client_id", "client_id"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    registered_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("registered_models.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    framework: Mapped[str] = mapped_column(String(80), nullable=False, default="pytorch")
    weights_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    training_seconds: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    registered_model: Mapped[RegisteredModel] = relationship(
        back_populates="versions", lazy="joined"
    )
    dataset_version: Mapped[DatasetVersion | None] = relationship(
        back_populates="model_versions", lazy="joined"
    )
    inference_logs: Mapped[list[InferenceLog]] = relationship(
        back_populates="model_version", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint(
            "registered_model_id", "model_version", name="uq_model_versions_model_version"
        ),
        Index("ix_model_versions_registered_model_id", "registered_model_id"),
        Index("ix_model_versions_model_version", "model_version"),
    )


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )
    request_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    environment: Mapped[InferenceEnvironment] = mapped_column(
        Enum(InferenceEnvironment, name="inference_environment"), nullable=False
    )
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_drift_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model_version: Mapped[ModelVersion] = relationship(
        back_populates="inference_logs", lazy="joined"
    )

    __table_args__ = (
        Index("ix_inference_logs_model_version_id", "model_version_id"),
        Index("ix_inference_logs_created_at", "created_at"),
        Index("ix_inference_logs_confidence_score", "confidence_score"),
    )


async def get_best_model_for_client_by_accuracy(
    session: AsyncSession, client_id: uuid.UUID
) -> ModelVersion | None:
    """Return the best model version for a client ordered by metrics['accuracy'].

    Notes:
    - Accuracy is extracted from JSONB metrics with a numeric cast.
    - selectinload() eagerly loads inference logs in a second efficient query,
      reducing N+1 issues when reading log aggregates afterwards.
    """

    accuracy_expr = ModelVersion.metrics["accuracy"].astext.cast(Numeric(10, 6))

    stmt = (
        select(ModelVersion)
        .join(ModelVersion.registered_model)
        .where(RegisteredModel.client_id == client_id)
        .where(ModelVersion.metrics["accuracy"].is_not(None))
        .options(selectinload(ModelVersion.inference_logs))
        .order_by(accuracy_expr.desc(), ModelVersion.created_at.desc())
        .limit(1)
    )

    result = await session.execute(stmt)
    return result.scalar_one_or_none()
