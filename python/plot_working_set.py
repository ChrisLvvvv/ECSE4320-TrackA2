import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_kv_line(line: str) -> dict:
    d = {}
    for part in line.strip().split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d

def load_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_kv_line(line))
    return pd.DataFrame(rows)

def main():
    in_path = "results/raw/cache_transition.csv"
    if not os.path.exists(in_path):
        print("missing", in_path)
        return

    os.makedirs("results/figures", exist_ok=True)

    df = load_file(in_path)

    # normalize columns
    for c in ["m", "k", "n", "threads", "time_s", "gflops", "density"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # infer 'size' for square cases
    if "m" in df.columns:
        df["size"] = df["m"]
    else:
        df["size"] = range(len(df))

    # keep only expected kernels
    df = df[df["kernel"].isin(["dense_gemm", "spmm_csr"])].copy()

    # plot GFLOP/s vs size
    plt.figure()
    for kernel in ["dense_gemm", "spmm_csr"]:
        sub = df[df["kernel"] == kernel].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("size")
        if "gflops" in sub.columns:
            plt.plot(sub["size"], sub["gflops"], marker="o", label=kernel)
    plt.xlabel("matrix size (m=k=n)")
    plt.ylabel("GFLOP/s")
    plt.title("Working-set transition: GFLOP/s vs size")
    plt.legend()
    out1 = "results/figures/cache_transition_gflops.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out1)

    # plot time vs size
    plt.figure()
    for kernel in ["dense_gemm", "spmm_csr"]:
        sub = df[df["kernel"] == kernel].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("size")
        if "time_s" in sub.columns:
            plt.plot(sub["size"], sub["time_s"], marker="o", label=kernel)
    plt.xlabel("matrix size (m=k=n)")
    plt.ylabel("time (s)")
    plt.title("Working-set transition: time vs size")
    plt.legend()
    out2 = "results/figures/cache_transition_time.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out2)

if __name__ == "__main__":
    main()
