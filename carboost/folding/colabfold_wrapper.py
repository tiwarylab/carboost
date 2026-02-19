"""
Utilities to run ColabFold predictions from Python code.

This module wraps the Colab notebook workflow into reusable functions so it can
be called from scripts or notebooks.
"""

from __future__ import annotations

import json
import re
import shutil
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .colabfold_utils import _import_colabfold, _make_prediction_callback, _make_input_features_callback, _prepare_result_folder, _add_hash, _check

@dataclass
class ColabFoldRunConfig:
    """A Class container to contain required configuration to run `colabfold`."""
    
    
    query_sequence: str
    jobname: str = "test42"
    model_type: str = "auto"
    max_msa: str = "8:16"
    num_seeds: int = 16
    msa_mode: str = "mmseqs2"
    result_dir: str | Path | None = None
    use_templates: bool = False
    custom_template_path: str | Path | None = None
    num_recycles: int = 1
    relax_max_iterations: int = 200
    recycle_early_stop_tolerance: float = None
    num_seeds: int = 1
    use_dropout: bool = True
    pairing_strategy: str = "greedy"
    pair_mode: str = "unpaired_paired"
    dpi: int = 300
    save_all: bool = False
    save_recycles: bool = False
    save_to_google_drive: bool = False
    num_models: int = 5
    model_order: tuple[int, ...] = (1, 2, 3, 4, 5)
    rank_by: str = "auto"
    stop_at_score: float = 100.0
    num_relax: int = 0
  
    

def run_colabfold_pipeline(config: ColabFoldRunConfig,
                           output_dir: str | Path = "structures",
                           *,
                           zip_output: bool = True,
                           overwrite: bool = True,
                           show_msa_plot: bool = False,
                           show_prediction_plot: bool = False,
                           ):
    """
    The main wrapper function that is based on colabfold notebook.
    This function collects MSA and runs colabfold to generated structures.
    Check the implementation in the demo notebooks.
    """
    query_sequence = "".join(config.query_sequence.split())
    basejobname = "".join(config.jobname.split())
    basejobname = re.sub(r'\W+', '', basejobname)

    jobname = _add_hash(basejobname, config.query_sequence)

    if not _check(jobname):
        n = 0
        while not _check(f"{jobname}_{n}"): n += 1
        jobname = f"{jobname}_{n}"
    os.makedirs(jobname, exist_ok=True)

    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
      text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    if "mmseqs2" in config.msa_mode:
      a3m_file = os.path.join(jobname,f"{jobname}.a3m")
    else:
        a3m_file = os.path.join(jobname,f"{jobname}.single_sequence.a3m")
    with open(a3m_file, "w") as text_file:
        text_file.write(">1\n%s" % query_sequence)

    funcs = _import_colabfold()
    result_dir = Path(jobname)
    log_filename = os.path.join(jobname,"log.txt")
    funcs["setup_logging"](Path(log_filename))


    queries, is_complex = funcs["get_queries"](queries_path)
    model_type = funcs["set_model_type"](is_complex, config.model_type)

    funcs = _import_colabfold()
    result_dir = Path(jobname)
    log_filename = os.path.join(jobname,"log.txt")
    funcs["setup_logging"](Path(log_filename))


    queries, is_complex = funcs["get_queries"](queries_path)
    model_type = funcs["set_model_type"](is_complex, config.model_type)

    if "multimer" in model_type and config.max_msa is not None:
        use_cluster_profile = False
    else:
        use_cluster_profile = True

    funcs["download_alphafold_params"](model_type, Path("."))

    prediction_callback = _make_prediction_callback(show_prediction_plot)
    input_features_callback = _make_input_features_callback(
        show_msa_plot, funcs["plot_msa_v2"]
    )

    results = funcs["run"](
            queries=queries,
            result_dir=result_dir,
            use_templates=config.use_templates,
            custom_template_path=config.custom_template_path,
            num_relax=config.num_relax,
            msa_mode=config.msa_mode,
            model_type=model_type,
            num_models=config.num_models,
            num_recycles=config.num_recycles,
            relax_max_iterations=config.relax_max_iterations,
            recycle_early_stop_tolerance=config.recycle_early_stop_tolerance,
            num_seeds=config.num_seeds,
            use_dropout=config.use_dropout,
            model_order=list(config.model_order),
            is_complex=is_complex,
            data_dir=Path("."),
            keep_existing_results=False,
            rank_by=config.rank_by,
            pair_mode=config.pair_mode,
            pairing_strategy=config.pairing_strategy,
            stop_at_score=float(config.stop_at_score),
            prediction_callback=prediction_callback,
            dpi=config.dpi,
            zip_results=False,
            save_all=config.save_all,
            max_msa=config.max_msa,
            use_cluster_profile=use_cluster_profile,
            input_features_callback=input_features_callback,
            save_recycles=config.save_recycles,
            user_agent="colabfold/google-colab-main",
        )

    export_dir = Path(output_dir)
    _prepare_result_folder(export_dir, overwrite=overwrite)

    jobname_prefix = ".custom" if config.msa_mode == "custom" else ""
    copied = []

    for idx in range(config.num_seeds * config.num_models):
        tag = results["rank"][0][idx]
        pdb_name = f"{jobname}{jobname_prefix}_unrelaxed_{tag}.pdb"
        json_name = f"{jobname}{jobname_prefix}_scores_{tag}.json"
        pdb_src = result_dir / pdb_name
        json_src = result_dir / json_name

        pdb_dst = export_dir / f"pred_{idx}.pdb"
        json_dst = export_dir / f"pred_{idx}.json"

        if pdb_src.exists():
            shutil.copy2(pdb_src, pdb_dst)
        if json_src.exists():
            shutil.copy2(json_src, json_dst)
        copied.append({"tag": tag, "pdb": str(pdb_dst), "json": str(json_dst)})

    config_json = result_dir / "config.json"
    msa_a3m = result_dir / f"{jobname}.a3m"
    if config_json.exists():
        shutil.copy2(config_json, export_dir / "config.json")
    if msa_a3m.exists():
        shutil.copy2(msa_a3m, export_dir / "msa.a3m")

    archive_path = None
    if zip_output:
        archive_path = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir)

    summary = {"result_dir": result_dir.resolve()._raw_paths[0],\
                "export_dir": export_dir.resolve()._raw_paths[0],\
                "zip_path": archive_path,\
                "copied_files": copied,}

    with open(export_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


    return summary