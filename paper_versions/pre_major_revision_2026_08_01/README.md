# Pre-revision backup — paper state as of 2026-08-01

This folder is a **frozen snapshot of the paper as it stood immediately before the major
revision** (5-page cut, claim corrections, float restructuring). It exists so the group can
restore or continue editing the *pre-revision* version from source if the revised version is
not wanted.

**Do not edit anything in this folder.** Treat it as read-only. The live paper is at the
repository root (`../../paper.tex`).

## Provenance

| | |
|---|---|
| Commit at time of backup | `619d382f137e74c8c28b0345c671cbff0d28731d` (`619d382`, "paper as a pdf") |
| Branch | `main` |
| Backup created | 2026-08-01 |
| Working tree state | Clean for all paper files. Five unrelated files had uncommitted line-ending-only (CRLF) diffs at the time: `requirements.txt`, `src/retrieval/pcst_retrieval.py`, `tests/pcst_fallback.py`, `tests/fixtures/mini_corpus.jsonl`, `tests/fixtures/mini_queries.jsonl`. None of these affect the paper. |
| PDF in this folder | 11 pages (main text runs to ~9.7 pages before references, plus a 1-page appendix) |

Every file here was byte-verified (MD5) against the repository copy at the moment of backup.

## Contents

```
README.md                  this file
paper.tex                  LaTeX source (558 lines)
custom.bib                 bibliography (8 entries)
graphs_project.pdf         the compiled PDF this source produces
generate_figures.py        script that produced the three figures
figures/
  fig_oracle_vs_recall.pdf / .png
  fig_pcst_sensitivity.pdf / .png
  fig_qtype_breakdown.pdf  / .png
```

`paper.tex` includes only the three `.pdf` figures via `\includegraphics`; the `.png` copies are
kept because they are tracked in the repository and are useful for slides/preview.

Deliberately **not** included: LaTeX build artifacts (`.aux`, `.log`, `.out`, `.bbl`, `.blg`,
`.synctex.gz`). None of them are needed — `graphs_project.pdf` is the compiled output, and the
bibliography is regenerated from `custom.bib` by BibTeX on any rebuild.

## ⚠️ Reproducibility gap you should close: the ACL style files are not in this repository

`paper.tex` begins with `\usepackage{acl}`, but **`acl.sty` is not present anywhere in this
repository**, and it is not part of a standard TeX Live installation. The current PDF was
therefore compiled somewhere that supplies it — almost certainly the Overleaf ACL 2023 template.

This means **this backup is fully reproducible only in that same Overleaf project**, not from a
bare clone of this repository. To make the snapshot genuinely self-contained, add these two
files from the official ACL style-files distribution
(<https://github.com/acl-org/acl-style-files>, `latex/` directory) into this folder and into the
repository root:

- `acl.sty`
- `acl_natbib.bst`

They could not be added automatically because this environment has no network access to fetch
them. Until they are added, restoring this version means re-uploading these files to Overleaf,
or copying them from the Overleaf project that currently compiles the paper.

## How to restore this version

**Option A — restore over the live paper (destructive to the revised version):**

```bash
cd <repo root>
cp paper_versions/pre_major_revision_2026_08_01/paper.tex        paper.tex
cp paper_versions/pre_major_revision_2026_08_01/custom.bib       custom.bib
cp paper_versions/pre_major_revision_2026_08_01/generate_figures.py generate_figures.py
cp paper_versions/pre_major_revision_2026_08_01/figures/*        figures/
cp paper_versions/pre_major_revision_2026_08_01/graphs_project.pdf graphs_project.pdf
```

**Option B — restore from git instead** (equivalent, and preferable if the backup commit is on
`origin`):

```bash
git checkout 619d382 -- paper.tex custom.bib figures/ generate_figures.py
```

**Option C — inspect only:** open `graphs_project.pdf` in this folder.

## Compiling

With `acl.sty` and `acl_natbib.bst` present in the same directory:

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

or `latexmk -pdf paper.tex`. There is no Makefile or `latexmkrc` in this repository.

## Regenerating the figures

`generate_figures.py` is self-contained with respect to data (the plotted values are hard-coded
in the script; it does not read the `experiment_handoff/` CSVs at run time). It does, however,
assume a Google Colab environment: it calls `google.colab.drive.mount()` and writes to
`/content/drive/MyDrive/graphrag_mknn_2026_07_30/figures/`. To run it outside Colab, remove the
`drive.mount` call and point `SAVE_DIR` at a local path.

Note that the figure values are hard-coded rather than read from `experiment_handoff/`, so if
the underlying results are ever re-run, the figures will **not** update automatically and must
be checked by hand against `experiment_handoff/`.
