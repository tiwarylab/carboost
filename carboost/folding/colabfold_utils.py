import re
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_colabfold():
    """ A function to get the required colabfold functions 
        while also checking if colabfold is installed"""

    try:
        from colabfold.batch import get_queries, run, set_model_type
        from colabfold.download import download_alphafold_params
        from colabfold.plot import plot_msa_v2
        from colabfold.utils import setup_logging
    except ImportError as exc:
        raise ImportError(
            "colabfold is not installed. Install it first, for example:\n"
            "pip install 'colabfold[alphafold-minus-jax] @ "
            "git+https://github.com/sokrypton/ColabFold'"
        ) from exc

    return {
        "download_alphafold_params": download_alphafold_params,
        "get_queries": get_queries,
        "plot_msa_v2": plot_msa_v2,
        "run": run,
        "set_model_type": set_model_type,
        "setup_logging": setup_logging,
    }


def _make_prediction_callback(enable_plot):
    """ Based on the colabfold notebook
    """
    if not enable_plot:
        return None

    import matplotlib.pyplot as plt
    from colabfold.colabfold import plot_protein

    def prediction_callback(protein_obj, length, prediction_result, input_features, mode):
        del prediction_result, input_features
        model_name, relaxed = mode
        del model_name, relaxed
        fig = plot_protein(protein_obj, Ls=length, dpi=150)
        plt.show()
        plt.close(fig)

    return prediction_callback


def _make_input_features_callback(enable_plot, plot_msa_v2):
    """ Based on the colabfold notebook
    """
    if not enable_plot:
        return None

    import matplotlib.pyplot as plt

    def input_features_callback(input_features):
        plot_msa_v2(input_features)
        plt.show()
        plt.close()

    return input_features_callback


def _prepare_result_folder(path: Path, overwrite):
    """ To create an output folder while checking previous file with that name exists.
    """
    if path.exists():
        if not overwrite:
            backup = Path('backup')
            print(f'FileExistsWarning: Output folder already exists: {path.resolve()._raw_paths[0]}\n\t\t   Files are moved to {backup.resolve()._raw_paths[0]}')
            os.system(f'mv {path.resolve()._raw_paths[0]} {backup.resolve()._raw_paths[0]}')
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def _add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

def _check(folder):
  if os.path.exists(folder):
    return False
  else:
    return True
