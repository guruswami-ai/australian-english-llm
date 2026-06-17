#!/usr/bin/env python3
"""
Stage 2: tokenize + pack cleaned AU corpus into fixed-length Gemma-2 sequences
for continued pre-training.

Each doc -> [bos] tokens [eos], concatenated and chunked into SEQ_LEN blocks.
Output parquet has a single column `input_ids` (list<int32>), ready for a
CPT trainer (labels = input_ids).

Usage: ./venv/bin/python tokenize_pack_gemma.py <in.parquet> <out.parquet> [seq_len]
"""
import sys, os, time
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B")
READ_BATCH = 20_000

def main():
    src = sys.argv[1]
    out = sys.argv[2]
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    writer = pq.ParquetWriter(out, schema, compression="zstd")

    pf = pq.ParquetFile(src)
    buf, seqs_chunk = [], []
    n_seq = n_tok = n_doc = 0
    t0 = time.time()

    def flush_rows():
        nonlocal seqs_chunk
        if seqs_chunk:
            writer.write_table(pa.table({"input_ids": seqs_chunk}, schema=schema))
            seqs_chunk = []

    for rb in pf.iter_batches(batch_size=READ_BATCH, columns=["text"]):
        texts = rb.column("text").to_pylist()
        n_doc += len(texts)
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if bos is not None:
                buf.append(bos)
            buf.extend(ids); buf.append(eos)
        # emit full blocks
        while len(buf) >= seq_len:
            seqs_chunk.append(buf[:seq_len])
            del buf[:seq_len]
            n_seq += 1; n_tok += seq_len
        if len(seqs_chunk) >= 2000:
            flush_rows()
        if n_doc % 200_000 < READ_BATCH:
            rate = n_doc / (time.time() - t0)
            print(f"  {n_doc:,} docs -> {n_seq:,} seqs | {rate:,.0f} docs/s", flush=True)
    flush_rows()
    writer.close()
    print(f"\nDone: {n_doc:,} docs -> {n_seq:,} seqs x {seq_len} = {n_tok/1e9:.2f}B tokens "
          f"in {(time.time()-t0)/60:.1f} min -> {os.path.basename(out)} "
          f"({os.path.getsize(out)/1e9:.1f} GB)")

if __name__ == "__main__":
    main()
