import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# 让 alembic 能 import 到 src.*（项目根加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# 两库分离后，**迁移必须对每个库各跑一次**（见 scripts/migrate_all.py）。
#
# ⚠️ ALEMBIC_DATABASE_URL **必须显式提供**（D9-②，fail-closed）：
#    早期这里写的是 `os.environ.get(...) or get_settings().database_url` —— 不设就回落真实库。
#    后果是裸跑 `alembic upgrade head` **只迁一个库、毫无提示**，之后压测库"莫名查询失败"
#    （正是本项目最贵的一类故障：静默降级）。现在不设就直接报错，并指向正确入口。
_db_url = os.environ.get("ALEMBIC_DATABASE_URL")
if not _db_url:
    raise RuntimeError(
        "未设置 ALEMBIC_DATABASE_URL —— 拒绝回落到某个默认库。\n"
        "两库分离后，迁移必须对每个库各跑一次，请改用统一入口：\n"
        "    python scripts/migrate_all.py             # 迁两个库 + 校验 schema 一致\n"
        "    python scripts/migrate_all.py --kb real   # 只迁真实库\n"
        "（确需手工单库迁移时，请自行显式导出 ALEMBIC_DATABASE_URL 后再跑 alembic）"
    )
config.set_main_option("sqlalchemy.url", _db_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
