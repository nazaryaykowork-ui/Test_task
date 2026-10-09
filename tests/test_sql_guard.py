"""The SQL guard: pure function, so these hold whatever the model writes."""

import re

import pytest
import sqlglot

from agent.sql_guard import SQLRejected, guard_sql

DATASET = "bigquery-public-data.thelook_ecommerce"
PII = ("first_name", "last_name", "email", "street_address", "postal_code", "latitude", "longitude", "user_geom")


def guard(sql: str, categories=("Jeans", "Pants")) -> str:
    return guard_sql(sql, DATASET, categories, PII)


def cte_sql(scoped: str, name: str) -> str:
    query = sqlglot.parse_one(scoped, read="bigquery")
    return next(c for c in query.args["with_"].expressions if c.alias == name).this.sql(dialect="bigquery")


def test_every_table_is_limited_to_the_users_categories():
    scoped = guard("SELECT COUNT(*) FROM orders")
    assert "category IN ('Jeans', 'Pants')" in cte_sql(scoped, "products")
    assert "JOIN products AS p" in cte_sql(scoped, "order_items")
    assert "SELECT order_id FROM order_items" in cte_sql(scoped, "orders")
    assert "SELECT user_id FROM order_items" in cte_sql(scoped, "users")


def test_users_cte_drops_every_pii_column():
    assert f"EXCEPT ({', '.join(PII)})" in cte_sql(guard("SELECT COUNT(*) FROM users"), "users")


def test_scope_ctes_come_before_the_models_own_ctes():
    scoped = guard("WITH top AS (SELECT user_id FROM order_items) SELECT COUNT(*) FROM top")
    names = [c.alias for c in sqlglot.parse_one(scoped, read="bigquery").args["with_"].expressions]
    assert names == ["products", "order_items", "orders", "users", "top"]


def test_union_queries_are_scoped():
    scoped = guard("SELECT id FROM products UNION ALL SELECT product_id FROM order_items")
    assert scoped.startswith("WITH products AS")


def test_category_names_are_escaped():
    assert r"'Men\'s'" in guard("SELECT 1 FROM products", categories=("Men's",))


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("SELECT 1; DROP TABLE orders", "exactly one"),
        ("DELETE FROM orders WHERE TRUE", "read-only SELECT"),
        ("INSERT INTO orders (order_id) VALUES (1)", "read-only SELECT"),
        ("CREATE TABLE x AS SELECT 1", "read-only SELECT"),
        ("SELEC id FROM orders", "could not be parsed"),
        (f"SELECT id FROM `{DATASET}.users`", "unqualified"),
        ("SELECT * FROM thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS", "unqualified"),
        ("SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')", "Table functions"),
        ("SELECT id FROM distribution_centers", "Unknown table"),
        (f"WITH users AS (SELECT * FROM `{DATASET}.users`) SELECT email FROM users", "Do not define CTEs"),
        ("SELECT email FROM users", "personal data"),
        ("SELECT id FROM users WHERE LOWER(first_name) = 'jane'", "personal data"),
        ("SELECT u.street_address FROM orders o JOIN users u ON u.id = o.user_id", "personal data"),
        ("SELECT * FROM users", "SELECT * is not allowed"),
        ("SELECT * FROM orders o JOIN users u ON u.id = o.user_id", "SELECT * is not allowed"),
        ("SELECT u.* FROM orders o JOIN users u ON u.id = o.user_id", "users.* is not allowed"),
        ("SELECT COUNT(*) FROM (SELECT * FROM users)", "SELECT * is not allowed"),
    ],
)
def test_unsafe_sql_is_rejected(sql, reason):
    with pytest.raises(SQLRejected, match=re.escape(reason)):
        guard(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM users",
        "SELECT * FROM orders",
        "SELECT o.* FROM orders o JOIN users u ON u.id = o.user_id",
        "SELECT state, COUNT(*) AS n FROM users GROUP BY state",
        "SELECT user_id, SUM(sale_price) AS spend FROM order_items GROUP BY user_id ORDER BY spend DESC LIMIT 10",
    ],
)
def test_safe_sql_is_allowed(sql):
    guard(sql)
