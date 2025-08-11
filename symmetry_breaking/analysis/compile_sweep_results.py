import pandas as pd
import json
from pathlib import Path

def compile_sweep_results(output_dir, pattern="*.json"):
    """
    Load all sweep output files from a directory into a single pandas DataFrame.

    Parameters
    ----------
    output_dir : str or Path
        Path to the directory containing the sweep output JSON files.
    pattern : str
        Glob pattern to match files (default: '*.json').

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per simulation and columns for params/metrics.
    """
    output_dir = Path(output_dir)
    rows = []

    for path in output_dir.glob(pattern):
        try:
            # First try pandas' JSON reader for Series format
            s = pd.read_json(path, typ="series")
            if isinstance(s, pd.DataFrame):
                # In case of odd shape, take first column
                s = s.iloc[:, 0]
            s = s.copy()
            s["__source_file__"] = path.name
            rows.append(s)
        except Exception:
            # Fallback: raw json -> Series
            try:
                with open(path, "r") as f:
                    obj = json.load(f)
                s = pd.Series(obj)
                s["__source_file__"] = path.name
                rows.append(s)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")

    if not rows:
        raise RuntimeError(f"No valid result files found in {output_dir} with pattern '{pattern}'")

    df = pd.DataFrame(rows)
    # Attempt numeric conversion
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

if __name__ == "__main__":
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    sweep_name = "sweep01"
    df = compile_sweep_results(output_dir=root / sweep_name)
    print(df.shape)
    print(df.head())
