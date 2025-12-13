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
    files = sorted(glob.glob("results/raw/cache_*.csv"))
    if not files:
        print("no cache_*.csv found in results/raw/ (ok for now)")
        return

    for path in files:
        df = load_file(path)
        df = to_numeric(df, ["m","k","n","time_s","gflops"])
        title = os.path.basename(path).replace(".csv","")

        # x-axis: problem size proxy
        if all(c in df.columns for c in ["m","k","n"]):
            df["size_proxy"] = df["m"] * df["k"] + df["k"] * df["n"] + df["m"] * df["n"]
        else:
            df["size_proxy"] = range(len(df))

        plt.figure()
        for kernel in df["kernel"].dropna().unique():
            sub = df[df["kernel"] == kernel].copy().sort_values("size_proxy")
            if sub.empty or "gflops" not in sub.columns:
                continue
            plt.plot(sub["size_proxy"], sub["gflops"], marker="o", label=kernel)

        plt.xlabel("size proxy (m*k + k*n + m*n)")
        plt.ylabel("GFLOP/s")
        plt.title(title)
        plt.legend()
        out = f"results/figures/{title}_cache_transitions.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("wrote", out)

if __name__ == "__main__":
    main()
