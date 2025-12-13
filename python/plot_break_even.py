import os, glob
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
    df = pd.DataFrame(rows)
    return df

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    os.makedirs("results/figures", exist_ok=True)

    files = sorted(glob.glob("results/raw/density_*.csv"))
    if not files:
        print("no density_*.csv found in results/raw/")
        return

    for path in files:
        df = load_file(path)
        df = to_numeric(df, ["m","k","n","threads","block","density","time_s","gflops"])
        dense = df[df["kernel"] == "dense_gemm"].copy()
        spmm  = df[df["kernel"] == "spmm_csr"].copy()

        if dense.empty or spmm.empty:
            print(f"skip {path} (missing dense or spmm)")
            continue

        dense_t = float(dense["time_s"].iloc[0])
        spmm = spmm.sort_values("density")

        # break-even: first density where spmm faster
        be = None
        for _, r in spmm.iterrows():
            if r["time_s"] < dense_t:
                be = float(r["density"])
                break

        plt.figure()
        plt.plot([float(dense["density"].iloc[0]) if "density" in dense.columns and pd.notna(dense["density"].iloc[0]) else 1.0],
                 [dense_t], marker="o")  # dummy point if you want
        plt.plot(spmm["density"], spmm["time_s"], marker="o", label="CSR SpMM")
        plt.axhline(dense_t, linestyle="--", label="Dense GEMM")

        title = os.path.basename(path).replace(".csv","")
        if be is not None:
            plt.axvline(be, linestyle=":", label=f"break-even ~ {be:.3g}")
        plt.xlabel("density (nnz / (m*k))")
        plt.ylabel("time (s)")
        plt.title(title)
        plt.legend()
        out = f"results/figures/{title}_break_even.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("wrote", out)

if __name__ == "__main__":
    main()
