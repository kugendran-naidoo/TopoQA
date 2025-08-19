#!/usr/bin/env python3
"""
Refactored k_mac_inference_model.py

- Wraps the whole pipeline in main()
- Exposes concurrency ("jobs") & DataLoader workers as CLI flags
- Adds robust device selection (CUDA / MPS / CPU)
- Normalizes I/O paths; validates inputs; better error messages
- Sorts & filters model lists deterministically
- Uses torch.no_grad(), model.eval(), and safe checkpoint loading
- Handles empty/failed graphs gracefully; writes results deterministically
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
import inspect

# Prefer the modern DataLoader import path for PyG
try:
    from torch_geometric.loader import DataLoader  # PyG >= 2.0
except Exception:
    from torch_geometric.data import DataLoader  # fallback

from torch_geometric.data import Batch, Data

from torch.utils.data import Dataset

# Project-local imports
from src.get_interface import interface_batch
from src.topo_feature import topo_fea
from src.node_fea_df import node_fea
from src.graph import create_graph
from src.proteingat import GNN_edge1_edgepooling


# ----------------------------- Utilities ----------------------------- #

def get_device(prefer: str = "auto") -> torch.device:
    """
    Choose the best available device.
    prefer: "auto" | "cuda" | "mps" | "cpu"
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA requested but not available. Falling back to MPS/CPU.")
    elif prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available. Falling back to CUDA/CPU.")
    elif prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_models_from_dir(d: Path, suffixes: Sequence[str] = (".pdb", ".cif")) -> List[str]:
    """Return sorted model basenames (without extension)."""
    names = set()
    for s in suffixes:
        for f in d.glob(f"*{s}"):
            names.add(f.stem)
    return sorted(names)


# ---------------------- Pipeline wrappers (I/O) ---------------------- #

def run_interface_extraction(complex_dir: Path, interface_dir: Path, jobs: int) -> None:
    """Compute interfaces for all complexes in complex_dir into interface_dir."""
    interface_batch(str(complex_dir), str(interface_dir), int(jobs))


def run_topology_features(model_names: Sequence[str], complex_dir: Path, topo_dir: Path,
                          interface_dir: Path, jobs: int) -> None:
    """Compute topological features in parallel."""
    def _topo_one(name: str) -> None:
        try:
            topo_fea(name, str(complex_dir), str(interface_dir), str(topo_dir))
        except Exception as e:
            print(f"[topo_fea] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky")(delayed(_topo_one)(m) for m in model_names)

def _resolve_structure_path(name: str, complex_dir: Path) -> Optional[Path]:
    for ext in (".pdb", ".cif", ".mmcif"):
        p = complex_dir / f"{name}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def run_node_features(model_names: Sequence[str], complex_dir: Path, topo_dir: Path,
                      interface_dir: Path, fea_dir: Path, jobs: int) -> None:
    """Calculate node features (basic + PH) in parallel."""
    # detect which constructor we have
    try:
        _sig = signature(node_fea)
        _nparams = len(_sig.parameters)
    except Exception:
        _nparams = None  # fall back to try/except per-call

    def _node_one(name: str) -> None:
        try:
            src = _resolve_structure_path(name, complex_dir)
            if src is None:
                print(f"[node_fea] skip {name}: no source PDB/mmCIF under {complex_dir}")
                return

            # Prefer passing FULL PATH if supported
            nf = None
            if _nparams == 3:
                # expected: node_fea(full_path, interface_dir, topo_dir)
                nf = node_fea(str(src), str(interface_dir), str(topo_dir))
            elif _nparams == 4:
                # expected: node_fea(name, pdb_dir, interface_dir, topo_dir)
                nf = node_fea(name, str(complex_dir), str(interface_dir), str(topo_dir))
            else:
                # unknown signature — try full path first, then (name, pdb_dir, ...)
                try:
                    nf = node_fea(str(src), str(interface_dir), str(topo_dir))
                except TypeError:
                    nf = node_fea(name, str(complex_dir), str(interface_dir), str(topo_dir))

            fea_df, _ = nf.calculate_fea()
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", np.nan, inplace=True)
            fea_df = fea_df.dropna()
            (fea_dir / f"{name}.csv").write_text(fea_df.to_csv(index=False))
        except Exception as e:
            print(f"[node_fea] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky")(delayed(_node_one)(m) for m in model_names)

def run_graph_construction(model_names: Sequence[str], fea_dir: Path, interface_dir: Path,
                           arr_cutoff: Sequence[str], graph_dir: Path,
                           complex_dir: Path, jobs: int) -> None:
    """Build PyG graphs and store to disk, then validate each saved graph."""

    def _validate_graph_file(name: str) -> bool:
        """
        Try to load the saved graph and check the basics the model expects.
        Returns True if usable; False otherwise (and removes the bad file).
        """
        # The create_graph() code typically writes <name>.pt
        candidates = [
            graph_dir / f"{name}.pt",
            graph_dir / f"{name}.pth",
            graph_dir / f"{name}.bin",
        ]
        p = next((c for c in candidates if c.exists() and c.stat().st_size > 0), None)
        if p is None:
            print(f"[validate] {name}: no graph file produced.")
            return False

        try:
            g = torch.load(p, map_location="cpu")
            # Some save paths store a list; take the first element.
            if isinstance(g, list):
                if len(g) == 0:
                    raise ValueError("empty list")
                g = g[0]

            # Minimal fields the model & PyG ops rely on:
            missing = []
            for attr in ("x", "edge_index", "edge_attr"):
                if not hasattr(g, attr) or getattr(g, attr) is None:
                    missing.append(attr)
            if missing:
                raise ValueError(f"missing fields: {','.join(missing)}")

            # Sanity on shapes/dtypes
            if g.edge_index.dim() != 2 or g.edge_index.size(0) != 2:
                raise ValueError(f"bad edge_index shape: {tuple(g.edge_index.shape)}")
            if g.edge_attr.dim() != 2:
                raise ValueError(f"bad edge_attr shape: {tuple(g.edge_attr.shape)}")
            if g.edge_index.dtype != torch.long:
                # Fixup common pitfall
                g.edge_index = g.edge_index.long()
            # Some old artifacts may have a stale 'batch' attribute set to None
            if hasattr(g, "batch") and g.batch is None:
                delattr(g, "batch")

            # If we changed anything, rewrite to keep the fixed version on disk
            torch.save(g, p)
            return True

        except Exception as e:
            print(f"[validate] {name}: invalid graph ({e}). Removing.")
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _graph_one(name: str) -> None:
        try:
            create_graph(name, str(fea_dir), str(interface_dir), arr_cutoff, str(graph_dir), str(complex_dir))
            ok = _validate_graph_file(name)
            if not ok:
                print(f"[create_graph] {name}: validation failed; graph removed.")
        except Exception as e:
            print(f"[create_graph] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky")(delayed(_graph_one)(m) for m in model_names)

# ------------------------ Dataset / Inference ------------------------ #

class GraphDataset(Dataset):
    """Loads pre-built torch_geometric Data objects saved by create_graph()."""
    def __init__(self, graph_dir: Path, model_names: Sequence[str]):
        self.paths = []
        for name in model_names:
            candidates = [
                graph_dir / f"{name}.pt",
                graph_dir / f"{name}.pth",
                graph_dir / f"{name}.bin",
            ]
            for c in candidates:
                if c.exists():
                    self.paths.append(c)
                    break
        if not self.paths:
            raise FileNotFoundError(f"No graph files found in {graph_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        item = torch.load(p, map_location="cpu")
        return item

from torch_geometric.data import Batch, Data

def pyg_collate(data_list):
    cleaned = []
    for d in data_list:
        if hasattr(d, "batch"):
            try: delattr(d, "batch")
            except Exception: d.batch = None

        # Basic validity: nodes & edges present
        nn = getattr(d, "num_nodes", None)
        if nn is None and getattr(d, "x", None) is not None:
            nn = d.x.size(0)
        ei = getattr(d, "edge_index", None)
        has_nodes = (nn is not None and int(nn) > 0)
        has_edges = (ei is not None and ei.numel() > 0)
        if has_nodes and has_edges:
            cleaned.append(d)
        else:
            print("[warn] dropping invalid graph in batch: nodes/edges missing")

    if not cleaned:
        # Prevent Batch.from_data_list([]) crash; return an empty single-graph
        empty = Data()
        empty.edge_index = torch.empty((2,0), dtype=torch.long)
        empty.x = torch.empty((0,0))
        return Batch.from_data_list([empty])

    return Batch.from_data_list(cleaned)

def multi_pyg_collate(samples):
    """
    Collate function that supports:
      - samples = [Data, Data, ...]  -> returns a single Batch
      - samples = [[Data_i0, ..., Data_iL-1], ...] -> returns [Batch_0, ..., Batch_L-1]
    """
    if not samples:
        return []

    first = samples[0]

    # Case A: each sample is a (list|tuple) of Data, length = num_net
    if isinstance(first, (list, tuple)):
        L = len(first)
        out = []
        for i in range(L):
            data_list_i = []
            for s in samples:
                d = s[i]
                # Remove stale .batch (PyG will re-create it)
                if hasattr(d, "batch"):
                    try:
                        delattr(d, "batch")
                    except Exception:
                        d.batch = None
                data_list_i.append(d)
            out.append(Batch.from_data_list(data_list_i))
        return out

    # Case B: single Data per sample
    return pyg_collate(samples)

def load_model(checkpoint_path: Path, device: torch.device):
    chk = torch.load(str(checkpoint_path), map_location="cpu")
    model = GNN_edge1_edgepooling("mean", num_net=1, edge_dim=11, heads=8)
    state = chk["state_dict"] if isinstance(chk, dict) and "state_dict" in chk else chk
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

def _ensure_batch_vector(b):
    if getattr(b, "batch", None) is not None:
        return b
    if hasattr(b, "ptr") and b.ptr is not None:
        ptr = b.ptr
        counts = ptr[1:] - ptr[:-1]
        device = ptr.device
        b.batch = torch.arange(len(counts), device=device).repeat_interleave(counts)
        return b
    # Fallback build
    n = None
    if hasattr(b, "num_nodes") and b.num_nodes is not None:
        n = int(b.num_nodes)
    elif hasattr(b, "x") and b.x is not None:
        n = int(b.x.size(0))
    elif hasattr(b, "edge_index") and b.edge_index is not None and b.edge_index.numel() > 0:
        n = int(b.edge_index.max().item() + 1)
    if n is None:
        raise RuntimeError("Cannot determine number of nodes to build 'batch' vector.")
    b.batch = torch.zeros(n, dtype=torch.long, device=b.edge_index.device if hasattr(b, "edge_index") else None)
    return b

def infer(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds: List[float] = []
    printed = False  # one-time debug print

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            # -- one-time debug (use first Batch if list) --
            if not printed:
                try:
                    n_batches = len(loader)
                except Exception:
                    n_batches = -1
                b0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                try:
                    n_nodes = int(b0.num_nodes) if hasattr(b0, "num_nodes") else -1
                except Exception:
                    n_nodes = -1
                has_batch_vec = hasattr(b0, "batch") and (b0.batch is not None)
                print(f"[debug] infer(): batches={n_batches}, device={device}, "
                      f"first_batch_nodes={n_nodes}, has_batch={has_batch_vec}")
                if hasattr(b0, "edge_index"):
                    try:
                        ei = b0.edge_index
                        print(f"[debug] first_batch edge_index shape: {tuple(ei.size())}")
                    except Exception as e:
                        print(f"[debug] edge_index shape unavailable: {e}")
                printed = True

            # -- move to device & ensure .batch vectors exist --
            if isinstance(batch, (list, tuple)):
                batch = [ _ensure_batch_vector(b.to(device)) for b in batch ]
                model_input = batch
            else:
                b = _ensure_batch_vector(batch.to(device))
                model_input = [b]   # model expects a list

            # -- forward --
            out = model(model_input)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.detach().cpu().numpy().ravel()
            preds.extend(out.tolist())

    return np.asarray(preds, dtype=np.float32)


# ------------------------------ main ------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoQA inference pipeline (refactored).")
    p.add_argument("--complex-folder", type=str, required=True,
                   help="Folder with input complex structures (.pdb/.cif).")
    p.add_argument("--work-dir", type=str, required=True,
                   help="Working directory for intermediate artifacts.")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory to write result CSV.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained GNN checkpoint (.pt/.pth).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (PyTorch). 0 = main process.")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                   help="Parallel jobs for feature/graph generation (joblib).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--cutoff", type=float, default=10.0,
                   help="Inter-chain Cα–Cα cutoff used by create_graph (Å).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute intermediates even if present.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"[info] device = {device}")

    complex_dir = Path(args.complex_folder).expanduser().resolve()
    work_dir    = Path(args.work_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    ckpt_path   = Path(args.checkpoint).expanduser().resolve()

    if not complex_dir.is_dir():
        print(f"[error] complex-folder does not exist: {complex_dir}")
        return 2
    if not any(complex_dir.glob("*.pdb")) and not any(complex_dir.glob("*.cif")):
        print(f"[error] no .pdb/.cif found in complex-folder: {complex_dir}")
        return 2
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return 2

    ensure_dir(work_dir)
    ensure_dir(results_dir)

    interface_dir = work_dir / "interface_ca"
    topo_dir      = work_dir / "node_topo"
    fea_dir       = work_dir / "node_fea"
    graph_dir     = work_dir / "graph"
    for d in (interface_dir, topo_dir, fea_dir, graph_dir):
        ensure_dir(d)

    # Determine model list deterministically
    all_models = list_models_from_dir(complex_dir)
    if not all_models:
        print(f"[error] No input complexes in {complex_dir}")
        return 2
    print(f"[info] found {len(all_models)} models in {complex_dir}")

    # Step 1: Interfaces
    if args.overwrite or not any(interface_dir.iterdir()):
        print(f"[stage] extracting interfaces -> {interface_dir} (jobs={args.jobs})")
        run_interface_extraction(complex_dir, interface_dir, args.jobs)
    else:
        print(f"[stage] reusing existing interfaces in {interface_dir}")

    # K - change
    # Keep models for which interface extraction succeeded
    iface_exts = (".txt", ".csv")  # <- your repo emits .txt here
    models_with_iface = sorted({
        p.stem
        for p in interface_dir.iterdir()
        if p.is_file() and p.suffix.lower() in iface_exts and p.stat().st_size > 0
})

    print(f"[debug] interface files found: {len(models_with_iface)}")
    if len(models_with_iface) == 0:
        print(f"[debug] sample of files in {interface_dir}:")
        for p in list(interface_dir.iterdir())[:10]:
            print(" -", p.name, p.stat().st_size, "bytes")

    model_names = sorted(set(all_models) & set(models_with_iface))
    if not model_names:
        print("[error] No models with detected interfaces. Nothing to do.")
        return 2
    print(f"[info] {len(model_names)} models with interfaces.")


    # K - change
    # ---- NEW robust validation 
    def _resolve_model_file(name: str, base: Path) -> Optional[Path]:
        """Return full path if a .pdb/.cif file exists for model `name`, else None."""
        for ext in (".pdb", ".cif"):
            candidate = base / f"{name}{ext}"
            if candidate.exists():
                return candidate
        return None

    missing = [m for m in model_names if _resolve_model_file(m, complex_dir) is None]
    if missing:
        print(f"[warn] {len(missing)} models skipped (no source PDB/mmCIF found): {missing[:5]} ...")
        model_names = [m for m in model_names if m not in missing]
    if not model_names:
        print("[error] No models left after validating source files.")
        return 2


    # Step 2: Topology features
    if args.overwrite or not any(topo_dir.iterdir()):
        print(f"[stage] computing topology features -> {topo_dir} (jobs={args.jobs})")
        run_topology_features(model_names, complex_dir, topo_dir, interface_dir, args.jobs)
    else:
        print(f"[stage] reusing existing topology in {topo_dir}")

    # Step 3: Node features
    if args.overwrite or not any(fea_dir.iterdir()):
        print(f"[stage] computing node features -> {fea_dir} (jobs={args.jobs})")
        run_node_features(model_names, complex_dir, topo_dir, interface_dir, fea_dir, args.jobs)
    else:
        print(f"[stage] reusing existing features in {fea_dir}")

    # Step 4: Graphs
    arr_cutoff = [f"0-{int(args.cutoff)}"]  # original API used ['0-10']
    if args.overwrite or not any(graph_dir.iterdir()):
        print(f"[stage] constructing graphs -> {graph_dir} (jobs={args.jobs}, cutoff={arr_cutoff[0]} Å)")
        run_graph_construction(model_names, fea_dir, interface_dir, arr_cutoff, graph_dir, complex_dir, args.jobs)

        # after run_graph_construction(...)
        good = list(Path(graph_dir).glob("*.pt")) + list(Path(graph_dir).glob("*.pth")) + list(Path(graph_dir).glob("*.bin"))
        print(f"[stage] graphs available for dataset: {len(good)}")

    else:
        print(f"[stage] reusing existing graphs in {graph_dir}")

    # Dataset / loader
    ds = GraphDataset(graph_dir, model_names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type != "cpu"), collate_fn=multi_pyg_collate)

    # Load model & infer
    print(f"[stage] loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)
    print(f"[stage] running inference on {len(ds)} graphs (batch_size={args.batch_size})")
    pred = infer(model, loader, device)

    # Results
    model_order = [p.stem for p in ds.paths]
    out_df = pd.DataFrame({"MODEL": model_order, "PRED_DOCKQ": pred})
    out_df = out_df.sort_values("PRED_DOCKQ", ascending=False).reset_index(drop=True)
    out_csv = results_dir / "result.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[done] wrote {len(out_df)} predictions to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

