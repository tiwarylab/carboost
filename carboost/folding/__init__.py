'''
CARBOOST: AlphaFold2 module to generate structures
'''

from .colabfold_wrapper import ColabFoldRunConfig, run_colabfold_pipeline

__all__ = ["ColabFoldRunConfig", "run_colabfold_pipeline"]
