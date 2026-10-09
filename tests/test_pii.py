"""Output masking."""

import pytest

from agent.pii import mask_pii


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Contact jane.doe+vip@example.co.uk today", "Contact [EMAIL] today"),
        ("Call (555) 123-4567", "Call [PHONE]"),
        ("Call 555-123-4567 or 555.123.4567", "Call [PHONE] or [PHONE]"),
        ("Call +1 555 123 4567", "Call [PHONE]"),
        ("Ships to 742 Evergreen Terrace Ave.", "Ships to [ADDRESS]"),
        ("Lives at 12 Main St in Springfield", "Lives at [ADDRESS] in Springfield"),
        ("Customer: Jane Doe spent the most", "Customer: [NAME] spent the most"),
        ("Customer name: Jane Doe", "Customer name: [NAME]"),
        ("Name: Jane", "Name: [NAME]"),
        ("Top buyer is Mrs. Jane Doe", "Top buyer is Mrs. [NAME]"),
    ],
)
def test_pii_is_masked(text, expected):
    assert mask_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Revenue was $1,234,567.89 in 2024-03, up 12.5% on 2023.",
        "User 12345 spent $4,210 across 37 orders.",
        "Calvin Klein jeans outsold Levi's in Texas and New York.",
        "Q1 report for Client Acme Corp",
        "Order 5551234567 shipped",  # an unformatted 10-digit number is not treated as a phone
        # Labels and titles that name products and brands, not people (seen in the catalogue).
        "Top product name: Diesel Shioner Jean sold 8 units",
        "Brand name: Calvin Klein",
        "Dr Martens boots and Dr. Comfort socks",
        "Customer: Texas had the highest average order value",
    ],
)
def test_business_text_is_left_alone(text):
    assert mask_pii(text) == text
