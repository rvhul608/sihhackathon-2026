"""Deterministic, label-aware parsing of clean English OCR output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ParsedLine:
    text: str
    confidence: float


FIELD_NAMES = (
    "mrp", "net_quantity", "manufacturing_date", "expiry_date", "manufacturer_name",
    "manufacturer_address", "consumer_care_contact", "unit_sale_price", "country_of_origin",
)


def parse_declarations(
    lines: list[ParsedLine], *, min_confidence: float = 0.0
) -> dict[str, dict[str, Any]]:
    """Return every contract field, preserving absence separately from confidence."""
    result = {name: _absent() for name in FIELD_NAMES}
    for index, line in enumerate(lines):
        text = line.text.strip()
        normalized = re.sub(r"\s+", " ", text)
        lowered = normalized.lower()

        _set_if_found(result, "mrp", _money_after_label(normalized, r"(?:m\.?(?:r\.?)?p\.?|maximum retail price)"), line.confidence, min_confidence)
        _set_if_found(result, "net_quantity", _quantity_after_label(normalized), line.confidence, min_confidence)
        _set_if_found(result, "manufacturing_date", _date_after_label(normalized, r"(?:mfg|mfd|manufactur(?:ed|ing)|packed)(?:\s*date)?"), line.confidence, min_confidence)
        _set_if_found(result, "expiry_date", _date_after_label(normalized, r"(?:exp(?:iry)?|use\s*by|best\s*before)(?:\s*date)?"), line.confidence, min_confidence)
        _set_if_found(result, "unit_sale_price", _money_after_label(normalized, r"(?:unit\s*(?:sale\s*)?price|price\s*per\s*(?:kg|g|l|ml))"), line.confidence, min_confidence)

        contact = _contact_after_label(normalized)
        if contact:
            _set_if_found(result, "consumer_care_contact", contact, line.confidence, min_confidence)
        origin = re.search(r"(?:country\s*of\s*origin|made\s*in)\s*[:\-]?\s*([A-Za-z ]{3,})", normalized, re.I)
        if origin:
            _set_if_found(result, "country_of_origin", origin.group(1).strip(" ."), line.confidence, min_confidence)

        if re.search(r"\b(?:manufactured\s+by|manufactured\s*&\s*marketed\s+by|marketed\s+by|packed\s+by)\b", lowered):
            value = re.split(r"\b(?:manufactured\s+by|manufactured\s*&\s*marketed\s+by|marketed\s+by|packed\s+by)\b\s*[:\-]?", normalized, flags=re.I)[-1].strip()
            if value:
                _set_if_found(result, "manufacturer_name", value, line.confidence, min_confidence)
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if next_line and _looks_like_address(next_line.text):
                _set_if_found(result, "manufacturer_address", next_line.text.strip(), next_line.confidence, min_confidence)
        elif re.search(r"\b(?:address|add\.)\b", lowered):
            address = re.split(r"\b(?:address|add\.)\b\s*[:\-]?", normalized, flags=re.I)[-1].strip()
            _set_if_found(result, "manufacturer_address", address, line.confidence, min_confidence)
    return result


def _absent() -> dict[str, Any]:
    return {"value": None, "confidence": 0.0, "present": False}


def _set_if_found(
    result: dict[str, dict[str, Any]], field: str, value: str | None,
    confidence: float, min_confidence: float = 0.0,
) -> None:
    if value and confidence >= min_confidence and not result[field]["present"]:
        result[field] = {"value": value, "confidence": round(max(0.0, min(1.0, confidence)), 2), "present": True}


def _money_after_label(text: str, label: str) -> str | None:
    match = re.search(label + r"\s*(?:\(?\s*(?:incl\.?\s*of\s*all\s*taxes?)?\s*\)?\s*)?[:\-]?\s*((?:₹|rs\.?|inr)\s*\d+(?:[,.]\d{1,2})?)", text, re.I)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def _quantity_after_label(text: str) -> str | None:
    match = re.search(r"(?:net\s*(?:qty|quantity|wt|weight|vol(?:ume)?)?)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:kg|g|mg|l|ml|litre?s?|pcs?|pieces?|nos?))", text, re.I)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def _date_after_label(text: str, label: str) -> str | None:
    match = re.search(label + r"\s*[:\-]?\s*((?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*[-/]?\s*\d{2,4})", text, re.I)
    return match.group(1).strip() if match else None


def _contact_after_label(text: str) -> str | None:
    if not re.search(r"(?:consumer|customer)\s*(?:care|service|helpline)|toll\s*free|contact", text, re.I):
        return None
    match = re.search(r"(?:\+?91[-\s]?)?(?:0?\d{3,5}[-\s]?)?\d{5,8}|\d{3,5}[-\s]\d{4,8}", text)
    return match.group(0).strip() if match else None


def _looks_like_address(text: str) -> bool:
    return bool(re.search(r"\d|(?:road|rd\.?|street|st\.?|nagar|district|india|pin)\b", text, re.I))
