"""Training pipelines for Gravion models.

Piece 7 ships the job specification and validation: a training
job names a catalog model and a dataset, and is checked against
the catalog and the module registry before anything runs.
Piece 11 adds dataset preparation (datasets.py): strict
validation, deduplication, a deterministic train/validation
split and a hashed dataset report.

Gravion Core also has a local training path: `run_training_job`
trains compact n-gram/TF-IDF weights from local Gravion dataset
files and saves a verifiable artifact. Specialist Gravion models
still require an operations-supplied weights bundle or endpoint;
their jobs remain validation-only in this build.

Mistake learning (`mistakes.py`) records failing tests, wrong
routes, weak responses and catalog failures, builds preference
pairs from corrected outputs, turns corrected cases into regression
tests, and folds corrected outputs plus verified catalog routing
examples back into local training.
"""
