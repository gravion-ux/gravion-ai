"""Training pipelines for Gravion models.

Piece 7 ships the job specification and validation: a training
job names a catalog model and a dataset, and is checked against
the catalog and the module registry before anything runs.
Actually training a model needs a training backend, which this
build does not contain — `run_training_job` says so plainly
instead of pretending. Piece 11 adds dataset preparation
(datasets.py): strict validation, deduplication, a deterministic
train/validation split and a hashed dataset report.
"""
