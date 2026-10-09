"""Output guard: masks PII in the final answer, as the last line of defence.

The first line is structural: PII columns can never be queried (agent/sql_guard.py), so
real customer data cannot reach the model. This mask catches what slips through anyway,
e.g. PII typed by the user and echoed back, or made up by the model. Names have no fixed
format, so only labelled names ("Name: ...", "Mr. ...") are masked; free-text person-name
detection needs NER (Cloud DLP in the production design).
"""

import re

_NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+)?"
_STREET = r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl"

PATTERNS = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[PHONE]", re.compile(r"(?<![\w$])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)")),
    ("[ADDRESS]", re.compile(rf"\b\d{{1,6}} (?:[A-Z][a-z]+ ){{1,3}}(?:{_STREET})\b\.?")),
    # Labelled names keep the label: "Customer name: Jane Doe" -> "Customer name: [NAME]".
    # A label right after product/brand/item/category/report names a thing, not a person.
    ("\\1[NAME]", re.compile(
        r"\b((?i:(?<!product )(?<!brand )(?<!item )(?<!category )(?<!report )"
        rf"(?:(?:customer|contact|full|first|last) )?name)\s*:\s*){_NAME}"
    )),
    # "Customer:" alone also starts sentences like "Customer: Texas had...", so it needs a
    # first and last name.
    ("\\1[NAME]", re.compile(r"\b((?i:customer)\s*:\s*)[A-Z][a-z]+ [A-Z][a-z]+\b")),
    # Period required, and no "Dr.": "Dr Martens", "Dr. Comfort" and "Dr. Rey" are brands in
    # the product catalogue.
    ("\\1[NAME]", re.compile(rf"\b((?:Mr|Mrs|Ms)\. ){_NAME}")),
]


def mask_pii(text: str) -> str:
    for replacement, pattern in PATTERNS:
        text = pattern.sub(replacement, text)
    return text
