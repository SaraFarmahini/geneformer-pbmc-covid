import os, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import BertForSequenceClassification

dev_mps = torch.device("mps") if torch.backends.mps.is_available() else None
print(f"mps available: {dev_mps is not None}", flush=True)
print(f"torch version: {torch.__version__}", flush=True)

M = "Geneformer/Geneformer-V1-10M"
print("loading model ...", flush=True)
t0 = time.time()
base = BertForSequenceClassification.from_pretrained(M, num_labels=2, output_hidden_states=False, output_attentions=False)
print(f"  loaded in {time.time()-t0:.1f}s; layers={base.config.num_hidden_layers}, hidden={base.config.hidden_size}, max_pos={base.config.max_position_embeddings}", flush=True)
vocab = base.config.vocab_size

def bench(device, batch, seqlen, n=3, train=True):
    import copy
    m = copy.deepcopy(base).to(device)
    m.train() if train else m.eval()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5) if train else None
    t = []
    for i in range(n + 1):
        x = torch.randint(0, vocab, (batch, seqlen), device=device)
        attn = torch.ones_like(x, device=device)
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        out = m(input_ids=x, attention_mask=attn, labels=labels)
        if train:
            out.loss.backward()
            opt.step(); opt.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        if i > 0:
            t.append(dt)
        sys.stdout.write(f"  {device.type} bs={batch} seq={seqlen} iter={i} dt={dt*1000:.0f}ms\n")
        sys.stdout.flush()
    del m
    if device.type == "mps":
        torch.mps.empty_cache()
    return sum(t) / len(t) if t else float("nan")

for dev_name, dev in [("mps", dev_mps), ("cpu", torch.device("cpu"))]:
    if dev is None: continue
    for seq, bs in [(512, 4), (1024, 4), (2048, 4), (1024, 2)]:
        try:
            dt = bench(dev, bs, seq, n=2, train=True)
            print(f"=> {dev_name}  bs={bs:>2d} seq={seq:>4d}  avg train step: {dt*1000:.0f} ms", flush=True)
        except Exception as e:
            print(f"=> {dev_name}  bs={bs:>2d} seq={seq:>4d}  FAIL: {type(e).__name__}: {str(e)[:80]}", flush=True)
