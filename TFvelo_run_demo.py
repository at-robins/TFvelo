import pandas as pd
import TFvelo as TFv
import anndata as ad
import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib

matplotlib.use("AGG")

import os, sys


def check_data_type(adata):
    for key in list(adata.var):
        if adata.var[key][0] in ["True", "False"]:
            adata.var[key] = adata.var[key].map({"True": True, "False": False})
    return


def data_type_tostr(adata, key):
    if key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            adata.var[key] = adata.var[key].map({True: "True", False: "False"})
    return


def preprocess(args):
    print(
        "----------------------------------preprocess ",
        args.dataset_path,
        "---------------------------------------------",
        flush=True,
    )
    adata = ad.read_h5ad(args.dataset_path)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata.uns["genes_all"] = np.array(adata.var_names)

    adata.X = adata.layers["counts"].copy()
    adata.layers["total"] = adata.X
    adata.layers["total_raw"] = adata.layers["total"].copy()
    n_cells, n_genes = adata.X.shape
    sc.pp.filter_genes(adata, min_cells=int(n_cells / 50))
    sc.pp.filter_cells(adata, min_genes=int(n_genes / 50))
    TFv.pp.filter_and_normalize(
        adata, min_shared_counts=20, n_top_genes=2000, log=True
    )  # include the following steps
    adata.X = adata.layers["total"].copy()

    adata.uns["clusters_colors"] = np.array(
        [
            "red",
            "orange",
            "yellow",
            "green",
            "skyblue",
            "blue",
            "purple",
            "pink",
            "#8fbc8f",
            "#f4a460",
            "#fdbf6f",
            "#ff7f00",
            "#b2df8a",
            "#1f78b4",
            "#6a3d9a",
            "#cab2d6",
        ],
        dtype=object,
    )

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    TFv.pp.moments(adata, n_pcs=30, n_neighbors=args.n_neighbors)

    TFv.pp.get_TFs(adata, databases=args.TF_databases)
    print(adata)
    adata.uns["genes_pp"] = np.array(adata.var_names)
    adata.write(os.path.join(args.result_path, "pp.h5ad"))


def main(args):
    print("--------------------------------")
    adata = ad.read_h5ad(os.path.join(args.result_path, "pp.h5ad"))

    print("n_jobs:", args.n_jobs, flush=True)
    flag = TFv.tl.recover_dynamics(
        adata,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
        var_names=args.var_names,
        WX_method=args.WX_method,
        WX_thres=args.WX_thres,
        max_n_TF=args.max_n_TF,
        n_top_genes=args.n_top_genes,
        fit_scaling=True,
        use_raw=args.use_raw,
        init_weight_method=args.init_weight_method,
        n_time_points=args.n_time_points,
    )
    if flag == False:
        return adata, False
    if "highly_variable_genes" in adata.var.keys():
        data_type_tostr(adata, key="highly_variable_genes")
    adata.write(os.path.join(args.result_path, "rc.h5ad"))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="pancreas",
        help="the dataset input path",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="number of cpus to use")
    parser.add_argument(
        "--var_names", type=str, default="all", help="all, highly_variable_genes"
    )
    parser.add_argument(
        "--init_weight_method",
        type=str,
        default="correlation",
        help="use correlation to initialize the weights",
    )
    parser.add_argument(
        "--WX_method",
        type=str,
        default="lsq_linear",
        help="LS, LASSO, Ridge, constant, LS_constrant, lsq_linear",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=30, help="number of neighbors"
    )
    parser.add_argument(
        "--WX_thres", type=int, default=20, help="the threshold for weights"
    )
    parser.add_argument("--n_top_genes", type=int, default=2000, help="n_top_genes")
    parser.add_argument(
        "--TF_databases", nargs="+", default="ENCODE ChEA", help="knockTF ChEA ENCODE"
    )
    parser.add_argument("--max_n_TF", type=int, default=99, help="max number of TFs")
    parser.add_argument(
        "--max_iter", type=int, default=20, help="max number of iteration in EM"
    )
    parser.add_argument("--n_time_points", type=int, default=1000, help="use_raw")
    parser.add_argument(
        "--result_path", type=str, default="_demo", help="output directory"
    )
    parser.add_argument("--use_raw", type=int, default=0, help="use_raw")
    parser.add_argument("--basis", type=str, default="umap", help="umap")

    args = parser.parse_args()
    print(
        "********************************************************************************************************"
    )
    print(args, flush=True)
    preprocess(args)
    main(args)
