"""
Add the role-scoped library table.

Revision ID: b1d4f6a83c20
Revises: c4e8a1b27d90
Create Date: 2026-09-30 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1d4f6a83c20"
down_revision: str | Sequence[str] | None = "c4e8a1b27d90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store named documents outside semantic features."""
    op.create_table(
        "library_entry",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("org_id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("role_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column(
            "always_loaded",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "org_id",
            "project_id",
            "role_id",
            "name",
            name="uq_library_entry_name",
        ),
    )


def downgrade() -> None:
    """Remove the library table."""
    op.drop_table("library_entry")
