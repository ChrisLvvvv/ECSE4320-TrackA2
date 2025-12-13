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
    return pd.DataFrame(rows)

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    os.makedirs("results/figures", exist_ok=True)
    files = sorted(glob.glob("results/raw/threads_*.csv"))
    if not files:
        print("no threads_*.csv found in results/raw/")
        return

    for path in files:
        df = load_file(path)
        df = to_numeric(df, ["threads","time_s","gflops","density","m","k","n"])
        title = os.path.basename(path).replace(".csv","")

        plt.figure()
        for kernel in ["dense_gemm", "spmm_csr"]:
            sub = df[df["kernel"] == kernel].copy().sort_values("threads")
            if sub.empty:
                continue
            base = float(sub["time_s"].iloc[0])
            speedup = base / sub["time_s"]
            plt.plot(sub["threads"], speedup, marker="o", label=kernel)

        plt.xlabel("threads")
        plt.ylabel("speedup (vs 1 thread)")
        plt.title(title)
        plt.legend()
        out = f"results/figures/{title}_speedup.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("wrote", out)

if __name__ == "__main__":
    main()
