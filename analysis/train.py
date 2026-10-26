"""Step 3b: train the autoencoder and save one embedding per sample.

Reads data/analysis/<mode>/samples.npz (from analysis.samples) and writes, in
the same folder:
    model.pt              the trained network
    embeddings.csv        sample_id, city, z1..zN (rotation-invariant)
    training_loss.csv/png how the error fell during training
    reconstructions.png   a few inputs next to the network's rebuilds

It runs on a normal laptop CPU (a few minutes for a few thousand tiles); a GPU
is used automatically if PyTorch can see one.

Usage:
    python -m analysis.train --mode tile
    python -m analysis.train --mode city --epochs 300 --latent 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .common import DATA_DIR, analysis_dir, say, write_json
from .model import ConvAutoencoder, d4


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(data_dir, mode, epochs=None, latent=16, batch=64, lr=1e-3, val_share=0.1,
          patience=20, seed=0, device="auto"):
    out = analysis_dir(data_dir, mode)
    if not (out / "samples.npz").exists():
        raise SystemExit(f"No samples in {out}. Run: python -m analysis.samples --mode {mode}")
    with np.load(out / "samples.npz") as z:
        x_np, channels = z["x"], list(z["channels"])
    meta = pd.read_csv(out / "samples.csv")
    n = x_np.shape[0]
    if epochs is None:
        epochs = 300 if mode == "city" else 60    # few city images need more passes
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)

    # Hold back some samples to check the network is not just memorising.
    order = rng.permutation(n)
    n_val = int(round(n * val_share)) if n >= 20 else 0
    val_idx, tr_idx = order[:n_val], order[n_val:]
    x = torch.from_numpy(x_np)
    model = ConvAutoencoder(x.shape[1], x.shape[2], latent=latent).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    say(f"Training on {len(tr_idx)} samples ({n_val} held back), {x.shape[1]} layers of "
        f"{x.shape[2]}x{x.shape[3]}, {epochs} epochs max, device {dev}")

    def batch_loss(idx, augment):
        xb = x[idx].to(dev)
        if augment:  # random rotation/flip, so north does not matter
            xb = d4(xb, int(rng.integers(8)))
        return torch.mean((model(xb) - xb) ** 2)

    history, best, best_state, since_best = [], np.inf, None, 0
    bs = min(batch, max(len(tr_idx), 1))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        tr_losses = []
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:   # batch norm needs at least 2 samples
                continue
            loss = batch_loss(idx, augment=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = np.mean([batch_loss(val_idx[i:i + bs], False).item()
                           for i in range(0, n_val, bs)]) if n_val else np.mean(tr_losses)
        history.append({"epoch": ep, "train_loss": np.mean(tr_losses), "val_loss": val})
        if val < best - 1e-6:
            best, since_best = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if ep == 1 or ep % 10 == 0:
            say(f"  epoch {ep:4d}  train {np.mean(tr_losses):.5f}  check {val:.5f}")
        if since_best >= patience:
            say(f"  stopped at epoch {ep}: no improvement for {patience} epochs")
            break
    model.load_state_dict(best_state)
    model.eval()
    say(f"  best check error {best:.5f} ({time.time() - t0:.0f} s)")

    # A baseline to judge the error against: always predicting the average image.
    baseline = float(torch.mean((x - x.mean(0, keepdim=True)) ** 2))
    say(f"  for comparison, predicting the average image gives {baseline:.5f}")

    with torch.no_grad():
        z = torch.cat([model.embed_invariant(x[i:i + 256].to(dev)).cpu()
                       for i in range(0, n, 256)]).numpy()
    emb = pd.concat([meta[["sample_id", "city"]],
                     pd.DataFrame(z, columns=[f"z{i + 1}" for i in range(z.shape[1])])], axis=1)
    emb.to_csv(out / "embeddings.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "in_ch": x.shape[1], "size": x.shape[2],
                "latent": latent, "channels": channels}, out / "model.pt")
    hist = pd.DataFrame(history)
    hist.to_csv(out / "training_loss.csv", index=False)
    write_json(out / "training_settings.json", {
        "epochs_run": len(history), "latent": latent, "batch": bs, "lr": lr, "seed": seed,
        "best_check_error": best, "average_image_error": baseline, "device": str(dev)})
    plots(out, hist, model, x, channels, dev, rng)
    say(f"  embeddings -> {out / 'embeddings.csv'}")
    return emb


def plots(out, hist, model, x, channels, dev, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist.epoch, hist.train_loss, label="training")
    ax.plot(hist.epoch, hist.val_loss, label="held-back check")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("rebuild error"); ax.legend()
    fig.savefig(out / "training_loss.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # Inputs vs rebuilds, summed over the POI layers (busiest examples first).
    poi = [i for i, c in enumerate(channels) if c.startswith("poi_")] or list(range(len(channels)))
    busy = torch.argsort(x[:, poi].sum((1, 2, 3)), descending=True)
    pick = busy[torch.linspace(0, len(busy) - 1, min(6, len(busy))).long()]
    with torch.no_grad():
        rec = model(x[pick].to(dev)).cpu()
    fig, axes = plt.subplots(2, len(pick), figsize=(2 * len(pick), 4.2), squeeze=False)
    for j in range(len(pick)):
        vmax = float(x[pick[j], poi].sum(0).max()) or 1
        axes[0, j].imshow(x[pick[j], poi].sum(0), cmap="magma", vmin=0, vmax=vmax)
        axes[1, j].imshow(rec[j, poi].sum(0), cmap="magma", vmin=0, vmax=vmax)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    axes[0, 0].set_title("input (POI layers)", fontsize=8, loc="left")
    axes[1, 0].set_title("network's rebuild", fontsize=8, loc="left")
    fig.savefig(out / "reconstructions.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def add_args(ap):
    ap.add_argument("--epochs", type=int, default=None, help="default 300 (city) / 60 (tile)")
    ap.add_argument("--latent", type=int, default=16, help="length of each embedding")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20, help="stop after this many epochs without improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--mode", choices=["city", "tile"], required=True)
    add_args(ap)
    a = ap.parse_args(argv)
    train(a.data_dir, a.mode, a.epochs, a.latent, a.batch, a.lr, patience=a.patience,
          seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
