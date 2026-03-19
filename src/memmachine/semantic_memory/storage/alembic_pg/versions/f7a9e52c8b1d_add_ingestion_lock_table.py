"""
add_ingestion_lock_table.

Revision ID: f7a9e52c8b1d
Revises: 62dff1150a46
Create Date: 2026-03-18 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7a9e52c8b1d"
down_revision: str | Sequence[str] | None = "62dff1150a46"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ingestion_lock table to prevent race conditions across pods."""
    op.create_table(
        "ingestion_lock",
        sa.Column("set_id", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column(
            "acquired_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("set_id"),
    )
    
    # Create index for cleanup queries
    op.create_index(
        "idx_ingestion_lock_expires_at",
        "ingestion_lock",
        ["expires_at"],
    )


def downgrade() -> None:
    """Remove ingestion_lock table."""
    op.drop_index("idx_ingestion_lock_expires_at", table_name="ingestion_lock")
    op.drop_table("ingestion_lock")
