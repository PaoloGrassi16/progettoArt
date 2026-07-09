from __future__ import annotations

import argparse
import io
from pathlib import Path


def add_github_link(input_pdf: Path, output_pdf: Path, github_url: str) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import ArrayObject, NumberObject
        from reportlab.lib.colors import HexColor
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError(
            "Install PDF dependencies with `pip install pypdf reportlab`."
        ) from exc

    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()
    first_page = reader.pages[0]
    width = float(first_page.mediabox.width)
    height = float(first_page.mediabox.height)

    packet = io.BytesIO()
    overlay = canvas.Canvas(packet, pagesize=(width, height))
    y = height - 340
    overlay.setFillColor(HexColor("#111827"))
    overlay.setFont("Helvetica-Bold", 9)
    overlay.drawString(50, y + 11, "Codice sorgente GitHub:")
    overlay.setFillColor(HexColor("#1d4ed8"))
    overlay.setFont("Helvetica", 9)
    overlay.drawString(168, y + 11, github_url)
    overlay.save()

    packet.seek(0)
    overlay_pdf = PdfReader(packet)
    first_page.merge_page(overlay_pdf.pages[0])

    for index, page in enumerate(reader.pages):
        writer.add_page(page)
        if index == 0:
            writer.add_uri(
                0,
                github_url,
                rect=(168, y + 8, min(width - 50, 168 + 330), y + 22),
                border=ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),
            )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a GitHub repository link to a PDF.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--github-url", required=True)
    args = parser.parse_args()
    add_github_link(args.input, args.output, args.github_url)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
