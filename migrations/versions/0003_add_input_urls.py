"""add image_url and video_url to video_jobs

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "video_jobs",
        sa.Column("image_url", sa.Text(), nullable=True),
    )
    op.add_column(
        "video_jobs",
        sa.Column("video_url", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("video_jobs", "video_url")
    op.drop_column("video_jobs", "image_url")
