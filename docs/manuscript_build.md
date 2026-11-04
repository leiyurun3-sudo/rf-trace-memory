# Extended manuscript

`paper/main.tex` and `paper/main.pdf` contain the public extended manuscript, including R86 and R87. The delivered PDF has 85 pages.

Build from `paper/` with Tectonic:

```powershell
New-Item -ItemType Directory -Force tmp/build | Out-Null
tectonic --keep-logs --outdir tmp/build main.tex
```

The publication sync was compiled successfully with Tectonic using cached resources. All 85 rebuilt pages matched the delivered PDF in extracted text and rendered pixels at 54 dpi. There were no unresolved citations/references or overfull boxes. The build retains the source's font-substitution and underfull-box warnings. All 23 TeX inputs in the dependency graph match the neutral source exactly; the delivered PDF is its unchanged byte copy.

Compact evidence is under `paper/evidence/`. R87 includes the frozen protocol, evaluated CSVs, statistics, the independent audit, 160 step summaries, 160 request-map manifests, and the 40 REQ response-map arrays. Original numerical values and statistical families are preserved: REQ–GLB raw p = 0.0061; REQ–PERM Holm-18 p = 0.0026.

Historical local paths in provenance and figure metadata are represented as repository-relative paths or `archive/...` identifiers. An archive identifier records provenance and does not imply that the external archive is bundled. Bibliographic venue names are retained.

The manuscript builds from its included figures. Optional figure regeneration can use `--r84` and `--r85` archive arguments in `build_semantic_preservation.py`; `build_final_qualitative.py` accepts the `R85_ARCHIVE` environment variable for the archive containing source images and previews. The optional recovered-label sensitivity script uses `QWEN_AUDIT_ROOT` for its external audit archive. These analysis and figure workflows were not rerun during publication sync.
