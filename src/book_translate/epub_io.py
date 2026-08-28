"""Locate the OPF package document and its NCX inside an EPUB.

Single source of truth: `META-INF/container.xml` names the OPF via
`rootfile@full-path`. Scanning for the first `*.opf` (rglob order is
filesystem-dependent) or guessing `content.opf` from a path prefix picks the
wrong file in books with multiple renditions or leftover/decoy `.opf` files.
The sorted scan is kept only as a fallback for broken books and always logs
a warning. The NCX is likewise resolved from the selected OPF (`spine@toc`
or the manifest item with the NCX media type), never by a global scan.
"""
from __future__ import annotations

import logging
import posixpath
import zipfile
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup

from ._util import decode_bytes, read_text

logger = logging.getLogger(__name__)

CONTAINER_PATH = "META-INF/container.xml"
OPF_MEDIA_TYPE = "application/oebps-package+xml"
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"


def _local(name: str | None) -> str:
    if not name:
        return ""
    return name.split("}")[-1]


def container_opf_path(container_xml: str) -> str | None:
    """`rootfile@full-path` from container.xml, via an XML parser (not regex).

    The first rootfile is the default rendition; entries declaring the OPF
    media type win over ones that omit it.
    """
    soup = BeautifulSoup(container_xml, "lxml-xml")
    rootfiles = [el for el in soup.find_all(True) if _local(el.name) == "rootfile"]
    declared = [el for el in rootfiles if (el.get("media-type") or "") == OPF_MEDIA_TYPE]
    for el in declared + rootfiles:
        path = (el.get("full-path") or "").strip().replace("\\", "/")
        if path:
            return posixpath.normpath(path).lstrip("/")
    return None


def _select_opf(names: set[str], read: Callable[[str], str], where: str) -> str | None:
    """Shared core for zip and extracted-tree lookup; `names` are posix paths."""
    declared = None
    if CONTAINER_PATH in names:
        try:
            declared = container_opf_path(read(CONTAINER_PATH))
        except Exception:
            declared = None
    if declared and declared in names:
        return declared
    if declared:
        logger.warning(
            "%s: %s points at missing OPF %r; falling back to *.opf scan",
            where,
            CONTAINER_PATH,
            declared,
        )
    fallback = sorted(n for n in names if n.lower().endswith(".opf"))
    if not fallback:
        return None
    if not declared:
        logger.warning(
            "%s: no usable %s; falling back to first of %d *.opf files",
            where,
            CONTAINER_PATH,
            len(fallback),
        )
    return fallback[0]


def find_opf_in_zip(z: zipfile.ZipFile) -> str | None:
    """Zip-internal path of the package OPF, or None."""
    names = set(z.namelist())
    where = getattr(z, "filename", None) or "epub"
    return _select_opf(names, lambda n: decode_bytes(z.read(n)), str(where))


def _tree_names(root: Path) -> set[str]:
    return {f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()}


def find_opf_in_tree(root: Path) -> Path | None:
    """Package OPF inside an extracted EPUB tree, or None."""
    root = Path(root)
    rel = _select_opf(_tree_names(root), lambda n: read_text(root / n), str(root))
    return root / rel if rel else None


def ncx_href_from_opf(opf_text: str) -> str | None:
    """NCX href declared by the OPF: `spine@toc` id first, then media type."""
    soup = BeautifulSoup(opf_text, "lxml-xml")
    hrefs: dict[str, str] = {}
    by_media: list[str] = []
    for el in soup.find_all(True):
        if _local(el.name) != "item":
            continue
        href = (el.get("href") or "").split("#", 1)[0].strip()
        if not href:
            continue
        ident = el.get("id")
        if ident:
            hrefs[str(ident)] = href
        if (el.get("media-type") or "") == NCX_MEDIA_TYPE:
            by_media.append(href)
    for el in soup.find_all(True):
        if _local(el.name) == "spine":
            toc_id = el.get("toc")
            if toc_id and str(toc_id) in hrefs:
                return hrefs[str(toc_id)]
            break
    return by_media[0] if by_media else None


def find_ncx_in_tree(root: Path, opf_path: Path | None) -> Path | None:
    """NCX file for the selected OPF inside an extracted EPUB tree.

    Only when there is no OPF at all does this fall back to a sorted scan
    (with a warning), so decoy `.ncx` files never shadow the declared one.
    """
    root = Path(root)
    if opf_path is not None:
        href = ncx_href_from_opf(read_text(opf_path))
        if not href:
            return None
        rel = posixpath.normpath(
            posixpath.join(opf_path.parent.relative_to(root).as_posix(), href)
        ).lstrip("/")
        cand = root / rel
        if cand.is_file():
            return cand
        logger.warning("%s: OPF declares missing NCX %r", root, rel)
        return None
    fallback = sorted(
        f for f in root.rglob("*") if f.is_file() and f.suffix.lower() == ".ncx"
    )
    if fallback:
        logger.warning(
            "%s: no OPF found; falling back to first of %d *.ncx files",
            root,
            len(fallback),
        )
        return fallback[0]
    return None
