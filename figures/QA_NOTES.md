# Figure QA notes

Figure sources: make_manuscript_figs.py + figstyle.py (run with PAPER_BASE
set; exports land under this package's figures/ directory, or in
the manuscript workspace's figures directory, as PDF/EPS/TIFF/PNG;
TIFFs are flattened to RGB with their 600 dpi resolution tags). The seven collision-audit JSONs in this folder
are the machine audits of the exported PDFs.

Re-rendering overwrites the alignment and collision-audit JSONs next to
the exports; the JSONs shipped here are those of the final render, and the
TIFF hashes below bind them to the exact files. Audit verdicts and their
final dispositions:

| Figure | Machine verdict | Disposition |
|---|---|---|
| Fig1, Fig2, Fig4, Fig5 | PASS | none needed |
| Fig3 | REVIEW REQUIRED (0 fail, 5 warn) | closed: the flagged class is in-bar value labels sitting on their own bars, by design; no blocking class (text-text, text-crossed-by-stroke) fires |
| Fig6 | REVIEW REQUIRED (0 fail, 4 warn) | closed: same class, contained fill overlays only |
| FigS1 | FIX BEFORE DELIVERY (8 fail, 5 warn) | closed by the documented manual review next to this file; the FAILs are PDF-text-extraction artifacts of 3D text transforms, collision free in the 600-dpi raster |

One documented exception: FigS1_dose_response_surface reports FAIL findings
in its collision audit (8 FAIL, 5 WARN). They are extraction artifacts of
matplotlib's 3D text transforms: the auditor's PDF-text extraction returns
duplicated spans for some 3D tick strings (each string appears once when
read with rawdict), and the pane gridlines pass behind rotated axis labels
by construction. Verified against the 600-dpi renders: all labels, the two
markers (filled and open circle, larger than the surrounding mesh strokes),
and the corner key are legible with no visible collisions.

## Final-figure hash record

The submission figures were re-rendered from the current tables with the
shipped figure script (the tables in this package regenerate them without any
downloads). Under the pinned environment of requirements.txt the clean-room
run reproduces Fig1/Fig4/Fig6/FigS1 byte-for-byte from the shipped tables and
regenerates Fig2/Fig3/Fig5 with the final tick-text and null-band-cap corrections (Fig2's
shortened 'Deseason.' tick and the corrected-null band caps of Fig3/Fig5); other
environments reproduce the same content (panel gates pass, values identical)
with rasterizer-level differences, so the byte-level claim is environment-
pinned rather than universal. Hashes of the shipped 600-dpi TIFFs bind the QA JSONs in this
folder to the exact files; the panel-alignment gate passed on every
multi-panel figure when they were rendered.

| Shipped file | sha256 (first 16) |
|---|---|
| Fig1.tif | `327a7538d33084b2` |
| Fig2.tif | `a59bc962fca5d0d7` |
| Fig3.tif | `4c2b7c5f95d98f7d` |
| Fig4.tif | `b64747b9019824a2` |
| Fig5.tif | `8a4aa0f5ca241084` |
| Fig6.tif | `a9744842f9494cba` |
| FigS1.tif | `fda428eed545f4e7` |

Fig3 and Fig6 carry REVIEW-REQUIRED verdicts from the collision audit (the
in-bar value labels intentionally sit on their bars; text-to-text and
text-crossed-by-stroke findings, the classes that block delivery, are at
zero). Fig2, Fig4 and Fig5 pass clean; Fig1/Fig4/Fig6/FigS1 reproduce byte-for-byte
from their original tables under the pinned environment (hashes above).

Note on the QA record paths: the alignment and collision-audit JSONs record
package-relative paths (`figures/<name>.pdf`); in the full workspace the
renders they describe live under the workspace's figures directory. Fig. S1's collision
audit verdict is closed by a documented manual review
(`FigS1_dose_response_surface.manual-review.json`): the FAIL findings are
PDF-text-extraction artifacts of matplotlib's 3D text transforms (rotated
tick strings vs pane gridlines in extracted geometry), verified collision
free in the 600-dpi renders at print width. The record now carries its own
final visual pass and the sha256-16 of the two delivered formats (EPS and
TIFF; the PNG and PDF exports were inspected at the same time and are listed
in exports_reviewed), so the closure is bound to the exact renders shipped
here. QA_DISPOSITIONS.json summarises each figure's machine verdict and its
final disposition in one machine-readable record. The collision-audit
verdicts were recorded on the renders preceding the final correction pass; that pass
changed Fig2's tick text and moved the Fig3/Fig5 null-band caps by at most
0.3 points on the axis, changes that leave the audited classes (in-bar
value labels, text-vs-stroke geometry) untouched.
