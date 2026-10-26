import argparse, glob, os, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--src", default="./data/kalm-finetune")
p.add_argument("--dst", default="./data/kalm-sample-1152k")
p.add_argument("--n", type=int, default=1_152_000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--exclude", nargs="*", default=[], help="task folder names to skip")
p.add_argument("--batch_rows", type=int, default=2000, help="rows streamed per batch")
a = p.parse_args()

files = []  # (task, path, nrows)
for d in sorted(glob.glob(os.path.join(a.src, "*/"))):
    task = os.path.basename(d.rstrip("/"))
    if task in a.exclude:
        continue
    for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
        files.append((task, f, pq.ParquetFile(f).metadata.num_rows))

total = sum(n for _, _, n in files)
assert a.n <= total, f"requested {a.n} > available {total}"
rng = np.random.default_rng(a.seed)
idx = np.sort(rng.choice(total, a.n, replace=False))

os.makedirs(a.dst, exist_ok=True)
manifest, off = {}, 0
for task, f, n in files:
    lo, hi = np.searchsorted(idx, [off, off + n])
    local = idx[lo:hi] - off
    off += n
    if len(local) == 0:
        continue
    pf = pq.ParquetFile(f)
    out_dir = os.path.join(a.dst, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(f))
    writer, pos, written = None, 0, 0
    # stream the file in small batches: avoids pyarrow's 2GB-per-chunk limit on big list<string> columns
    for batch in pf.iter_batches(batch_size=a.batch_rows):
        nb = batch.num_rows
        l, h = np.searchsorted(local, [pos, pos + nb])
        if h > l:
            sel = pa.Table.from_batches([batch], schema=pf.schema_arrow).take(pa.array(local[l:h] - pos))
            if writer is None:
                writer = pq.ParquetWriter(out_path, pf.schema_arrow)
            writer.write_table(sel)
            written += len(sel)
        pos += nb
    if writer is not None:
        writer.close()
    assert written == len(local), (f, written, len(local))
    manifest[task] = manifest.get(task, 0) + int(written)

json.dump({"seed": a.seed, "n": a.n, "total_available": total,
           "excluded": a.exclude, "per_task": manifest},
          open(os.path.join(a.dst, "manifest.json"), "w"), indent=1)
print(f"wrote {sum(manifest.values()):,} rows across {len(manifest)} tasks to {a.dst}")
