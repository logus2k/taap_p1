#!/usr/bin/env python3
"""
Convert Jupyter notebooks or Markdown documents to DOCX.
For notebooks: strip text outputs while keeping charts (images), then export to DOCX.
For Markdown: directly convert to DOCX.

Usage (CLI):
  python ipynb_or_md_to_docx.py path/to/input_file [options]

Usage (API):
  from ipynb_or_md_to_docx import DocumentConverter
  converter = DocumentConverter(template="my_template.docx", hide_code=True)
  result = converter.convert("my_notebook.ipynb")
  print(result.docx)  # Path to generated DOCX

Input formats supported:
  - Jupyter notebooks (.ipynb): text outputs stripped by default, images preserved
  - Markdown documents (.md): converted directly to DOCX

Requirements:
  pip install nbformat nbconvert
  System: pandoc (apt install pandoc)
  For SVG images in DOCX:
    Linux: sudo apt install librsvg2-bin

Options:
  --out report_basename    Base name for outputs (without extension)
  --template path.docx     Reference DOCX template for styling (page layout, fonts, etc.)
  --no-code                (Notebooks only) Hide code cells in the exported outputs
  --keep-text              (Notebooks only) Preserve text outputs from code cells
  --html                   (Notebooks only) Also export to HTML format
  --markdown               (Notebooks only) Also export to Markdown format

Examples:
  python ipynb_or_md_to_docx.py my_notebook.ipynb
  python ipynb_or_md_to_docx.py my_notebook.ipynb --no-code --keep-text
  python ipynb_or_md_to_docx.py my_document.md --template my_template.docx
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_HAS_NB = False

try:
    import nbformat as nbf  # type: ignore[import-untyped]
    from nbconvert import HTMLExporter, MarkdownExporter  # type: ignore[import-untyped]
    from traitlets.config import Config  # type: ignore[import-untyped]

    _HAS_NB = True
except ImportError:
    pass

if TYPE_CHECKING:
    import nbformat as nbf  # type: ignore[no-redef]
    from nbconvert import HTMLExporter, MarkdownExporter  # type: ignore[no-redef]
    from traitlets.config import Config  # type: ignore[no-redef]


class ConversionError(Exception):
    """Raised when a conversion step fails."""


@dataclass
class ConversionResult:
    """Paths to generated output files."""

    docx: Path | None = None
    html: Path | None = None
    markdown: Path | None = None


@dataclass
class DocumentConverter:
    """Converts Jupyter notebooks or Markdown files to DOCX (and optionally HTML/Markdown).

    Args:
        template: Path to a reference DOCX template for styling.
        hide_code: If True, exclude code cell inputs from notebook exports.
        keep_text: If True, preserve text outputs from code cells (default strips them).
        export_html: If True, also export notebooks to HTML.
        export_markdown: If True, also keep the intermediate Markdown export.
    """

    template: Path | None = None
    hide_code: bool = False
    keep_text: bool = False
    export_html: bool = False
    export_markdown: bool = False

    def __post_init__(self) -> None:
        if self.template is not None:
            self.template = Path(self.template).resolve()
            if not self.template.exists():
                raise ConversionError(f"Template file not found: {self.template}")

    def convert(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
        base_name: str | None = None,
    ) -> ConversionResult:
        """Convert an .ipynb or .md file to DOCX.

        Args:
            input_path: Path to the source file.
            output_dir: Directory for outputs. Defaults to the input file's parent.
            base_name: Base name for output files (no extension). Defaults to input stem.

        Returns:
            ConversionResult with paths to generated files.

        Raises:
            ConversionError: On invalid input, missing dependencies, or conversion failure.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise ConversionError(f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix not in (".ipynb", ".md"):
            raise ConversionError(
                f"Unsupported file type: {suffix}. Supported: .ipynb, .md"
            )

        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            base_name = input_path.stem

        if suffix == ".ipynb":
            return self._process_notebook(input_path, base_name, output_dir)
        else:
            return self._process_markdown(input_path, base_name, output_dir)

    # -- Private methods -------------------------------------------------------

    @staticmethod
    def _require_nb() -> None:
        """Ensure nbformat/nbconvert are available."""
        if not _HAS_NB:
            raise ConversionError(
                "nbformat/nbconvert not available. "
                "Install with: pip install nbformat nbconvert"
            )

    @staticmethod
    def _require_pandoc() -> None:
        """Ensure system pandoc is available."""
        if shutil.which("pandoc") is None:
            raise ConversionError(
                "pandoc not found on PATH. Install with: apt install pandoc"
            )

    @staticmethod
    def _clean_outputs(nb: nbf.NotebookNode, keep_text: bool = False) -> nbf.NotebookNode:
        """Process notebook cell outputs.

        Always clears execution_count. When keep_text is False, strips text-only
        outputs and keeps only image (PNG/SVG) and rich HTML outputs.
        """
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            if not keep_text:
                new_outputs = []
                for out in cell.get("outputs", []):
                    otype = out.get("output_type")
                    data = out.get("data", {})
                    has_img = any(
                        k in data for k in ("image/png", "image/svg+xml")
                    )
                    has_html = "text/html" in data
                    if otype in {"display_data", "execute_result"} and (
                        has_img or has_html
                    ):
                        new_outputs.append(out)
                cell["outputs"] = new_outputs
            cell["execution_count"] = None
        return nb

    def _export_html(self, nb: nbf.NotebookNode, out_path: Path) -> None:
        """Export notebook to HTML."""
        self._require_nb()
        c = Config()
        c.HTMLExporter.exclude_input_prompt = True
        c.HTMLExporter.exclude_output_prompt = True
        if self.hide_code:
            c.HTMLExporter.exclude_input = True
        html_exporter = HTMLExporter(config=c)
        body, _ = html_exporter.from_notebook_node(nb)
        out_path.write_text(body, encoding="utf-8")

    def _export_markdown(
        self, nb: nbf.NotebookNode, out_path: Path, assets_dir: str
    ) -> None:
        """Export notebook to Markdown, extracting images to assets_dir."""
        self._require_nb()
        c = Config()
        c.MarkdownExporter.exclude_input_prompt = True
        c.MarkdownExporter.exclude_output_prompt = True
        if self.hide_code:
            c.MarkdownExporter.exclude_input = True
        resources = {"output_files_dir": assets_dir}
        md_exporter = MarkdownExporter(config=c)
        body, resources = md_exporter.from_notebook_node(nb, resources=resources)
        out_path.write_text(body, encoding="utf-8")

        for fname, data in resources.get("outputs", {}).items():
            img_path = out_path.parent / fname
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(data)

    def _markdown_to_docx(self, md_path: Path, docx_path: Path) -> None:
        """Convert Markdown to DOCX via system pandoc."""
        self._require_pandoc()
        resource_dir = str(md_path.parent)
        cmd = [
            "pandoc",
            str(md_path),
            "-o",
            str(docx_path),
            "--standalone",
            f"--resource-path={resource_dir}",
        ]
        if self.template:
            cmd.append(f"--reference-doc={self.template}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ConversionError(
                f"pandoc failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    def _process_notebook(
        self, notebook_path: Path, base_name: str, output_dir: Path
    ) -> ConversionResult:
        """Process a Jupyter notebook file."""
        self._require_nb()

        out_html = output_dir / f"{base_name}_report.html"
        out_md = output_dir / f"{base_name}_report.md"
        assets_dir = f"{base_name}_report_files"
        out_docx = output_dir / f"{base_name}_report.docx"

        nb = nbf.read(str(notebook_path), as_version=4)
        nb_clean = self._clean_outputs(nb, keep_text=self.keep_text)

        result = ConversionResult()

        if self.export_html:
            self._export_html(nb_clean, out_html)
            result.html = out_html

        # Markdown is always needed as intermediate for DOCX
        self._export_markdown(nb_clean, out_md, assets_dir=assets_dir)

        if self.export_markdown:
            result.markdown = out_md

        self._markdown_to_docx(out_md, out_docx)
        result.docx = out_docx

        # Clean up intermediate files if markdown export was not requested
        if not self.export_markdown:
            out_md.unlink(missing_ok=True)
            assets_path = output_dir / assets_dir
            if assets_path.exists():
                shutil.rmtree(assets_path)

        return result

    def _process_markdown(
        self, markdown_path: Path, base_name: str, output_dir: Path
    ) -> ConversionResult:
        """Process a Markdown file directly."""
        out_docx = output_dir / f"{base_name}.docx"
        self._markdown_to_docx(markdown_path, out_docx)
        return ConversionResult(docx=out_docx)


# -- CLI entry point ----------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Jupyter notebooks or Markdown documents to DOCX."
    )
    ap.add_argument("input_file", help="Path to source .ipynb or .md file")
    ap.add_argument("--out", help="Base name for outputs (without extension)")
    ap.add_argument(
        "--template",
        help="Path to reference DOCX template for styling (page layout, fonts, etc.)",
    )
    ap.add_argument(
        "--no-code",
        action="store_true",
        help="(Notebooks only) Hide code cells in the exported outputs.",
    )
    ap.add_argument(
        "--keep-text",
        action="store_true",
        help="(Notebooks only) Preserve text outputs from code cells.",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="(Notebooks only) Also export to HTML format.",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="(Notebooks only) Also export to Markdown format.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_file).resolve()
    suffix = input_path.suffix.lower()

    if suffix == ".md" and any([args.no_code, args.html, args.markdown, args.keep_text]):
        print("[warning] Options --no-code, --html, --markdown, --keep-text are ignored for Markdown input")

    try:
        converter = DocumentConverter(
            template=args.template,
            hide_code=args.no_code,
            keep_text=args.keep_text,
            export_html=args.html,
            export_markdown=args.markdown,
        )
        result = converter.convert(input_path, base_name=args.out)
    except ConversionError as e:
        print(f"[error] {e}")
        raise SystemExit(1)

    print("\nDone. Outputs:")
    if result.html:
        print(f"  HTML report  : {result.html}")
    if result.markdown:
        print(f"  Markdown     : {result.markdown}")
    if result.docx:
        print(f"  Word (.docx) : {result.docx}")


if __name__ == "__main__":
    main()
