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

def main():
    os.makedirs("results/figures", exist_ok=True)
    files = sorted(glob.glob("results/raw/simd_vs_threads_*.csv"))
    if not files:
        print("no simd_vs_threads_*.csv found in results/raw/")
        return

    for path in files:
        df = load_file(path)

        for c in ["threads","time_s","gflops"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        title = os.path.basename(path).replace(".csv","")

        # Create labels for the 3 configurations:
        # 1) novec + 1 thread  -> "scalar"
        # 2) novec + T threads -> "threads"
        # 3) vec + T threads   -> "simd+threads"
        def label_row(r):
            if r.get("variant","") == "novec" and int(r.get("threads",0)) == 1:
                return "scalar (1T)"
            if r.get("variant","") == "novec" and int(r.get("threads",0)) > 1:
                return "threads (no-vec)"
            if r.get("variant","") == "vec" and int(r.get("threads",0)) > 1:
                return "SIMD+threads"
            return "other"

        df["mode"] = df.apply(label_row, axis=1)
        df = df[df["mode"].isin(["scalar (1T)","threads (no-vec)","SIMD+threads"])].copy()

        # Plot per kernel
        kernels = df["kernel"].unique()
        for kernel in kernels:
            sub = df[df["kernel"] == kernel].copy()
            order = ["scalar (1T)","threads (no-vec)","SIMD+threads"]
            sub["mode"] = pd.Categorical(sub["mode"], categories=order, ordered=True)
            sub = sub.sort_values("mode")

            if sub.empty:
                continue

            base = float(sub[sub["mode"]=="scalar (1T)"]["time_s"].iloc[0])
            sub["speedup_vs_scalar"] = base / sub["time_s"]

            plt.figure()
            plt.bar(sub["mode"], sub["speedup_vs_scalar"])
            plt.ylabel("speedup vs scalar (1 thread)")
            plt.title(f"{title} | {kernel}")
            out = f"results/figures/{title}_{kernel}_simd_vs_threads.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print("wrote", out)

if __name__ == "__main__":
    main()
