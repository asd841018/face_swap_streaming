"""create video_jobs table

Revision ID: 0001
Revises:
Create Date: 2026-04-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIDEO_JOB_STATUS_VALUES = (
    "queued",
    "downloading",
    "processing",
    "uploading",
    "completed",
    "failed",
)


def _status_enum() -> postgresql.ENUM:
    return postgresql.ENUM(
        *VIDEO_JOB_STATUS_VALUES,
        name="video_job_status",
        create_type=False,
    )


def upgrade() -> None:
    status_enum = _status_enum()
    status_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "video_jobs",
        sa.Column("job_id", sa.String(length=32), primary_key=True),
        sa.Column("owner_key", sa.String(length=128), nullable=False),
        sa.Column("status", status_enum, nullable=False, server_default="queued"),
        sa.Column("output_url", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_video_jobs_owner_key", "video_jobs", ["owner_key"])
    op.create_index("ix_video_jobs_status", "video_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_video_jobs_status", table_name="video_jobs")
    op.drop_index("ix_video_jobs_owner_key", table_name="video_jobs")
    op.drop_table("video_jobs")
    _status_enum().drop(op.get_bind(), checkfirst=True)
