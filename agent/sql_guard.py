"""Whatever the model writes, it can only read the user's product scope and never PII.

The model writes SQL against four unqualified names: orders, order_items, products, users.
The guard checks the query, then prepends CTEs with those names, built from the real tables
but filtered to the user's categories and without PII columns. "The WITH clause hides any
permanent tables with the same name" (BigQuery docs), so the model never reads a real table.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

TABLES = ("orders", "order_items", "products", "users")
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command)


class SQLRejected(ValueError):
    """The message goes back to the model."""


def guard_sql(sql: str, dataset: str, categories: tuple[str, ...], pii_columns: tuple[str, ...]) -> str:
    query = _parse_single_select(sql)
    _check_tables(query)
    _check_columns(query, {c.lower() for c in pii_columns})
    _check_no_star_on_users(query)
    return _with_scope(query, dataset, categories, pii_columns).sql(dialect="bigquery")


def _scope_ctes(dataset: str, categories: tuple[str, ...], pii_columns: tuple[str, ...]) -> str:
    """Users and orders are limited to those with at least one in-scope item."""
    in_scope = ", ".join(exp.Literal.string(c).sql(dialect="bigquery") for c in categories)
    return f"""
        products AS (SELECT * FROM `{dataset}.products` WHERE category IN ({in_scope})),
        order_items AS (SELECT oi.* FROM `{dataset}.order_items` AS oi JOIN products AS p ON oi.product_id = p.id),
        orders AS (SELECT * FROM `{dataset}.orders` WHERE order_id IN (SELECT order_id FROM order_items)),
        users AS (SELECT * EXCEPT ({", ".join(pii_columns)}) FROM `{dataset}.users`
                  WHERE id IN (SELECT user_id FROM order_items))
    """


def _parse_single_select(sql: str) -> exp.Query:
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except SqlglotError as e:
        raise SQLRejected(f"The SQL could not be parsed: {e}") from e
    if len(statements) != 1:
        raise SQLRejected("Write exactly one SQL statement.")
    query = statements[0]
    if not isinstance(query, exp.Query) or query.find(*FORBIDDEN):
        raise SQLRejected("Only read-only SELECT queries are allowed.")
    return query


def _check_tables(query: exp.Query) -> None:
    own_ctes = {cte.alias_or_name.lower() for cte in query.find_all(exp.CTE)}
    if shadowed := own_ctes & set(TABLES):
        raise SQLRejected(f"Do not define CTEs named {', '.join(sorted(shadowed))}; use another name.")
    for table in query.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLRejected("Table functions are not allowed; query the tables directly.")
        if table.db or table.catalog:
            raise SQLRejected(f"Use the unqualified table names only: {', '.join(TABLES)}.")
        if table.name.lower() not in TABLES and table.name.lower() not in own_ctes:
            raise SQLRejected(f"Unknown table '{table.name}'. Available tables: {', '.join(TABLES)}.")


def _check_columns(query: exp.Query, pii: set[str]) -> None:
    for column in query.find_all(exp.Column):
        if column.name.lower() in pii:
            raise SQLRejected(f"Column '{column.name}' is personal data and is not available. Use user id instead.")


def _check_no_star_on_users(query: exp.Query) -> None:
    """COUNT(*) is fine."""
    for select in query.find_all(exp.Select):
        sources = [select.args.get("from_")] + (select.args.get("joins") or [])
        aliases = {
            s.this.alias_or_name.lower(): s.this.name.lower()
            for s in sources
            if s is not None and isinstance(s.this, exp.Table)
        }
        for projection in select.expressions:
            if isinstance(projection, exp.Star) and "users" in aliases.values():
                raise SQLRejected("SELECT * is not allowed on users; list the columns you need.")
            if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                if aliases.get(projection.table.lower()) == "users":
                    raise SQLRejected("users.* is not allowed; list the columns you need.")


def _with_scope(query: exp.Query, dataset: str, categories: tuple[str, ...], pii_columns: tuple[str, ...]) -> exp.Query:
    """Puts the scope CTEs first in the query's WITH list, so its own CTEs can use them."""
    scope = sqlglot.parse_one(f"WITH {_scope_ctes(dataset, categories, pii_columns)} SELECT 1", read="bigquery")
    if existing := query.args.get("with_"):
        existing.set("expressions", scope.args["with_"].expressions + existing.expressions)
    else:
        query.set("with_", scope.args["with_"])
    return query
