# QA-CINN + CI-MoE (code)

```
pip install -r requirements.txt
python tests_numpy.py                                   # torch-free checks (features, quality, fusion, Prop. 1)
python run_all.py --synthetic                           # end-to-end software test (NOT results)
python run_all.py --manifest data/manifest.csv          # real data -> ../results/cimoe_results.tex + results.json
```
`manifest.csv` columns: `file,modality,label,specimen` (modality = ftir|raman; replicates of one physical specimen
share `specimen`; each file is a two-column wavenumber,intensity CSV). Then recompile `../main.tex` (Overleaf: upload
the whole project; the `results/` tables are `\input` automatically).

* All CI-MoE hyper-parameters are in `qacinn/config.py` and are fixed a priori (never tuned on held-out specimens).
* Folds/seeds/features are identical across models, so comparisons are paired.
* Raman LOSO with single-specimen classes leaves the held-out class unseen in training; the fold summaries record
  `unseen_classes` so this can be reported.
