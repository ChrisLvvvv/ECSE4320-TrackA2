import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_kv(line: str) -> dict:
    d = {}
    for part in line.strip().split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d

def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_kv(line))
    return pd.DataFrame(rows)

def ai_dense(m,k,n):
    flops = 2.0*m*k*n
    bytes_ = 8.0*(m*k + k*n + 2.0*m*n)
    return flops/bytes_

def ai_spmm(m,k,n,nnz):
    flops = 2.0*nnz*n
    bytes_ = 8.0*nnz + 4.0*nnz + 8.0*(m+1) + 8.0*nnz*n + 8.0*m*n
    return flops/bytes_

def main():
    in_path = "results/raw/roofline_points.csv"
    if not os.path.exists(in_path):
        print("missing", in_path)
        return

    os.makedirs("results/figures", exist_ok=True)

    df = load(in_path)

    # numeric columns if present
    for c in ["m","k","n","nnz","gflops","time_s","gbps","density","threads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bw = df[df["kernel"] == "stream_triad"]["gbps"].dropna()
    if bw.empty:
        print("missing stream_triad gbps")
        return
    bw_gbps = float(bw.max())  # sustained bandwidth estimate

    dense = df[df["kernel"] == "dense_gemm"].copy()
    spmm  = df[df["kernel"] == "spmm_csr"].copy()

    # compute AI
    if not dense.empty:
        dense["AI"] = dense.apply(lambda r: ai_dense(r["m"], r["k"], r["n"]), axis=1)
    if not spmm.empty:
        spmm["AI"]  = spmm.apply(lambda r: ai_spmm(r["m"], r["k"], r["n"], r["nnz"]), axis=1)

    # conservative peak: use best dense point as a lower-bound-ish "achievable peak"
    # and inflate slightly to form the roof (still conservative).
    peak = None
    if not dense.empty and "gflops" in dense.columns:
        best_dense = float(dense["gflops"].max())
        peak = best_dense * 1.15
    else:
        peak = 100.0

    # build roofline curve
    xs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    ys = [min(peak, bw_gbps * x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label=f"roofline (bw={bw_gbps:.1f} GB/s, peak≈{peak:.1f} GF/s)")
    plt.xscale("log")
    plt.yscale("log")

    # scatter points
    if not dense.empty:
        plt.scatter(dense["AI"], dense["gflops"], label="dense_gemm")
    if not spmm.empty:
        plt.scatter(spmm["AI"], spmm["gflops"], label="spmm_csr")

    plt.xlabel("arithmetic intensity (FLOP/byte)")
    plt.ylabel("GFLOP/s")
    plt.title("Roofline (measured bandwidth + achieved points)")
    plt.legend()

    out = "results/figures/roofline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out)

if __name__ == "__main__":
    main()
