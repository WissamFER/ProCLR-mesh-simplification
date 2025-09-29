#!/usr/bin/env python3
import os
import sys
import argparse
import traceback

def main():
    parser = argparse.ArgumentParser(
        description="Build a centroid-level graph from a .off mesh and save it as a pickle."
    )
    parser.add_argument("off_file_path", help="Path to the input .off mesh file.")
    parser.add_argument("properties_dir", help="Directory containing per-mesh property files named <basename>.txt.")
    parser.add_argument("output_pkl_path", help="Output path for the resulting NetworkX graph pickle (.pkl).")

    # Optional hyperparameters
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Geodesic radius for local patch extraction (default: 3.0).")
    parser.add_argument("--dist-th", type=float, default=0.5,
                        help="Distance threshold used by AgglomerativeClustering (default: 0.5).")
    parser.add_argument("--linkage", type=str, default="average",
                        choices=["average", "single", "complete", "ward"],
                        help="Linkage strategy for AgglomerativeClustering (default: average).")
    parser.add_argument("--max-k", type=int, default=10,
                        help="Maximum k to try when building the centroid k-NN graph (default: 10).")
    parser.add_argument("--feat-from", type=int, default=3,
                        help="Column index where feature space starts (default: 3, i.e., after xyz).")

    args = parser.parse_args()

   


    if not os.path.isfile(args.off_file_path):
        print(f"[ERROR] .off file not found: {args.off_file_path}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.properties_dir):
        print(f"[ERROR] Properties directory not found: {args.properties_dir}", file=sys.stderr)
        sys.exit(2)

    # Import here so environment errors surface clearly
    try:
        from mesh_simlification_to_graph import SingleCentroidGraph
    except Exception as e:
        print("[ERROR] Could not import SingleCentroidGraph from 'mesh_simlification_to_graph'.", file=sys.stderr)
        print(f"Cause: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)

    # Build centroid graph
    try:
        builder = SingleCentroidGraph(
            mesh_path=args.off_file_path,
            properties_dir=args.properties_dir,
            radius=args.radius,
            dist_th=args.dist_th,
            linkage=args.linkage,
            max_k=args.max_k,
            feat_from=args.feat_from,
        )
    except Exception as e:
        print("[ERROR] Failed to build centroid-level graph.", file=sys.stderr)
        print(f"Cause: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(4)

    # Save pickle
    try:
        out_dir = os.path.dirname(os.path.abspath(args.output_pkl_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        builder.save_pickle(args.output_pkl_path)
    except Exception as e:
        print("[ERROR] Failed to save pickle.", file=sys.stderr)
        print(f"Cause: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(5)

    # Summary
    G = builder.graph
    C = builder.centroids
    print("[OK] Pickle saved:", args.output_pkl_path)
    #print(f"    Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Centroids: {len(C)}")

if __name__ == "__main__":
    main()
