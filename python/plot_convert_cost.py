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

def load_kv_csv(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_kv_line(line))
    return pd.DataFrame(rows)

def main():
    os.makedirs("results/figures", exist_ok=True)

    files = sorted(glob.glob("results/raw/convert_m*_k*_n*_t*.csv"))
    if not files:
        print("no convert_*.csv found in results/raw/")
        return

    # plot the most recent one
    path = files[-1]
    df = load_kv_csv(path)

    for c in ["density", "time_s", "p95_s", "threads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    conv = df[df["kernel"] == "convert_csr"].copy()
    spmm = df[df["kernel"] == "spmm_csr"].copy()

    # align by density
    merged = pd.merge(conv, spmm, on="density", suffixes=("_conv", "_spmm"))
    merged = merged.sort_values("density")

    if merged.empty:
        print("no overlapping densities found")
        return

    # amortization: how many SpMM multiplies to pay for conversion
    merged["amortize_multiplies"] = merged["time_s_conv"] / merged["time_s_spmm"]

    title = os.path.basename(path).replace(".csv", "")
    plt.figure()
    plt.plot(merged["density"], merged["time_s_conv"], marker="o", label="dense→CSR conversion")
    plt.plot(merged["density"], merged["time_s_spmm"], marker="o", label="one CSR SpMM")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("density (nnz / (m*k))")
    plt.ylabel("time (s) [log scale]")
    plt.title(f"Format conversion cost: {title}")

    # annotate amortization at a few points
    for _, r in merged.iterrows():
        d = r["density"]
        if d in (0.001, 0.01, 0.05, 0.2):
            txt = f"{r['amortize_multiplies']:.1f}×"
            plt.annotate(txt, (d, r["time_s_conv"]), textcoords="offset points", xytext=(6, 6))

    plt.legend()
    out = "results/figures/convert_cost.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out)

if __name__ == "__main__":
    main()
