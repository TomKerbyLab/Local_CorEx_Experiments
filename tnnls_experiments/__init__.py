"""Local CorEx TNNLS experiment program.

Implements the experiment specifications in agent_instructions/. All experiments
write to a shared outputs/ directory at the repository root:

    outputs/
      configs/      resolved machine-readable configs for every run
      raw/          tidy raw results (Parquet/CSV/NPZ) keyed by run id
      summaries/    aggregate tables produced by summarizers
      figures/      generated figures
      logs/         run logs and failure records
      checkpoints/  neural checkpoints
"""

__version__ = "0.1.0"

# Experiment names used as run-id prefixes.
GAUSSIAN = "gaussian"
NONLINEAR = "nonlinear"
REDUNDANCY = "redundancy"
MNIST = "mnist"
IMAGENET = "imagenet"
DROPOUT = "dropout"