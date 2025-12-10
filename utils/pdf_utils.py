"""Utilities for generating PDF documents from markdown content."""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Iterable

from fpdf import FPDF
from fpdf.errors import FPDFException


logger = logging.getLogger(__name__)


def markdown_to_pdf_bytes(markdown_text: str, *, title: str | None = None) -> bytes:
    """Convert markdown content to a simple PDF document.

    The conversion preserves headings and list markers while otherwise treating the
    markdown as plain text. The result is returned as raw PDF bytes suitable for
    download in Streamlit or saving to disk.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    effective_width = getattr(pdf, "epw", pdf.w - pdf.l_margin - pdf.r_margin)

    if title:
        pdf.set_font("Helvetica", "B", 16)
        _write_cell(pdf, effective_width, 10, _latin1(title))
        pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    for line in _wrap_lines(_sanitize_markdown(markdown_text)):
        text = line if line.strip() else " "
        safe_text = _latin1(text)
        if not _write_cell(pdf, effective_width, 6, safe_text):
            logger.warning("Falling back to defensive wrapping in PDF generation")
            for chunk in _defensive_wrap(safe_text):
                _write_cell(pdf, effective_width, 6, chunk or " ")

    return _finalize_pdf_output(pdf)


def _write_cell(pdf: FPDF, width: float, height: float, text: str) -> bool:
    """Write a text cell safely; return False if it failed."""
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(width, height, text)
        return True
    except FPDFException as err:
        logger.error("FPDF multi_cell failed: %s", err)
        return False


def _defensive_wrap(text: str, chunk_size: int = 80) -> Iterable[str]:
    """Fallback wrapper that breaks text into fixed chunks."""
    if not text:
        yield ""
        return

    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _finalize_pdf_output(pdf: FPDF) -> bytes:
    """Return PDF bytes compatible with multiple fpdf versions."""
    try:
        raw_output = pdf.output()
    except TypeError:
        # Older fpdf versions require dest="S"
        raw_output = pdf.output(dest="S")

    if isinstance(raw_output, bytearray):
        return bytes(raw_output)

    if isinstance(raw_output, bytes):
        return raw_output

    if isinstance(raw_output, str):
        return raw_output.encode("latin-1")

    # Fallback to string conversion
    return str(raw_output).encode("latin-1", errors="ignore")


def _sanitize_markdown(markdown_text: str) -> Iterable[str]:
    """Normalize markdown into plain text lines."""
    lines = markdown_text.splitlines()
    in_code_block = False

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            yield raw_line
            continue

        if not stripped:
            yield ""
            continue

        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            heading = _remove_inline_markdown(heading).upper()
            yield heading
            yield ""
            continue

        if stripped.startswith(('- ', '* ', '+ ')):
            content = stripped[2:].strip()
            content = _remove_inline_markdown(content)
            yield f"- {content}"
            continue

        ordered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ordered_match:
            number, content = ordered_match.groups()
            content = _remove_inline_markdown(content.strip())
            yield f"{number}. {content}"
            continue

        if stripped.startswith(">"):
            quote_content = stripped.lstrip('>').strip()
            quote_content = _remove_inline_markdown(quote_content)
            yield f"> {quote_content}"
            continue

        plain = _remove_inline_markdown(stripped)
        yield plain


def _remove_inline_markdown(text: str) -> str:
    """Remove inline markdown markers while keeping the content."""
    # Remove markdown formatting while preserving the text content
    # Order matters: process bold before single asterisk/underscore
    cleaned = text
    
    # Bold: **text** or __text__
    cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'__(.+?)__', r'\1', cleaned)
    
    # Italic: *text* or _text_
    cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)
    cleaned = re.sub(r'_(.+?)_', r'\1', cleaned)
    
    # Code: `text`
    cleaned = re.sub(r'`(.+?)`', r'\1', cleaned)
    
    # Links: [text](url) -> text
    cleaned = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', cleaned)
    
    # Images: ![alt](url) -> alt
    cleaned = re.sub(r'!\[(.+?)\]\(.+?\)', r'\1', cleaned)
    
    return cleaned


def _wrap_lines(lines: Iterable[str], width: int = 95) -> Iterable[str]:
    """Wrap lines to keep PDF text readable."""
    for line in lines:
        if not line:
            yield ""
            continue

        wrapped = textwrap.wrap(line, width=width)
        if not wrapped:
            yield ""
            continue
        for segment in wrapped:
            yield segment


def _latin1(text: str) -> str:
    """Ensure text is latin-1 encodable for FPDF."""
    return text.encode("latin-1", errors="ignore").decode("latin-1")
