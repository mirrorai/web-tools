"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision}
Create Date: ${create_date}
"""

"""
Short how-to:
    * Rename table:
        op.rename_table('old_name', 'new_name')
    * Rename column:
        op.alter_column(
            'table',
            'old_name',
            nullable=ORIGINAL,
            new_column_name='new_name',
            existing_type=COLUMN_TYPE
        )
    * Nullability change
        op.alter_column('table', 'column', existing_type=COLUMN_TYPE, nullable=NEW_VALUE)

        WARN: when setting to nullable=False, need to deal with false values in table. One of the variants:

        op.execute('UPDATE table SET {0} = "{1}" WHERE {0} is NULL'.format(
            column, new_value
        ))
"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy_utils
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}


def upgrade():
    % if upgrades:
    ${upgrades}
    connection = op.get_bind()
    session = sa.orm.sessionmaker()(bind=connection)

    # TODO: Add object based actions here or remove
    # WARN: Use it like: session.query(User) NOT User.query

    session.commit()
    % else:
    pass
    % endif


def downgrade():
    % if downgrades:
    ${downgrades}
    connection = op.get_bind()
    session = sa.orm.sessionmaker()(bind=connection)

    # TODO: Add object based actions here or remove
    # WARN: Use it like: session.query(User) NOT User.query

    session.commit()
    % else:
    pass
    % endif
