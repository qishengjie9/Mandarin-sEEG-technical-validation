# GitHub release checklist

Complete this checklist before making the repository public.

- [ ] Confirm the repository title and dataset title used in the manuscript.
- [ ] Replace the temporary research-group copyright holder in `LICENSE` with
      the institutionally approved code copyright holder, or select another
      approved software licence.
- [ ] Replace every bracketed field in the DUA template.
- [ ] Obtain final ethics, legal, information-security, and institutional data
      governance approval for the DUA and access workflow.
- [ ] Add the final Data Descriptor citation and controlled-repository DOI.
- [ ] Add approved access-request and security contact addresses.
- [ ] Confirm the final software licence and institutional code owner also
      cover the decoding modules supplied with this repository.
- [ ] Confirm the public repository contains no NPY, NPZ, WAV, EDF, clinical,
      electrode-coordinate, participant, or credential files.
- [ ] Confirm no checkpoints, held-out prediction archives, trial manifests,
      cache files, or logs containing local paths are included.
- [ ] Inspect the complete Git history, not only the current working tree, for
      accidentally committed restricted data.
- [ ] Re-run `python -m compileall -q src tests` and `pytest` in a clean
      environment.
- [ ] Verify all six decoding script entry points return a working `--help`
      interface after `pip install -e ".[decoding]"`.
- [ ] Reproduce the reported figures using an authorized local copy of the
      controlled dataset.

