"""Add the role-scoped key-value log table.

Revision ID: c4e8a1b27d90
Revises: f7a9e52c8b1d
Create Date: 2026-09-30 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e8a1b27d90"
down_revision: str | Sequence[str] | None = "f7a9e52c8b1d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store append-only key-value rows outside semantic features."""
    op.create_table(
        "kv_entry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("org_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("role_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_kv_entry_lookup",
        "kv_entry",
        ["org_id", "project_id", "role_id", "key", "created_at"],
    )


def downgrade() -> None:
    """Remove the key-value log table."""
    op.drop_index("idx_kv_entry_lookup", table_name="kv_entry")
    op.drop_table("kv_entry")
