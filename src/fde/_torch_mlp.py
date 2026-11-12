"""Training of the PyTorch MLP member (nn_mlp), and its forward pass in numpy.

This module imports only numpy and PyTorch. On macOS, PyTorch and LightGBM/XGBoost each bring their own OpenMP
library, and training the MLP in a process where both are loaded crashes (segmentation fault). There the MLP is
therefore trained in a separate process that runs this module (`python -m fde._torch_mlp in.npz out.npz`), and it
predicts with `forward`, which needs numpy only. On other systems `train` runs in the calling process.
"""
import sys

import numpy as np


def train(A, ya, H, yh, K, seed, threads, dev="cpu"):
    """Fit the MLP (three ReLU layers of 256 units, dropout 0.1, Adam) with early stopping on (H, yh); returns the
    network on the CPU, in evaluation mode."""
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed)
    layers, w = [], A.shape[1]
    for _ in range(3):
        layers += [torch.nn.Linear(w, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1)]; w = 256
    net = torch.nn.Sequential(*layers, torch.nn.Linear(w, K))
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
    lossf, evalf = torch.nn.CrossEntropyLoss(label_smoothing=0.0), torch.nn.CrossEntropyLoss()
    At, yt = torch.tensor(A, dtype=torch.float32, device=dev), torch.tensor(ya, dtype=torch.long, device=dev)
    Ht, yht = torch.tensor(H, dtype=torch.float32, device=dev), torch.tensor(yh, dtype=torch.long, device=dev)
    best, state, wait, g = 1e18, None, 0, torch.Generator().manual_seed(seed)
    for _ in range(200):
        net.train()
        for idx in torch.randperm(len(At), generator=g).split(512):
            opt.zero_grad(); lossf(net(At[idx]), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(evalf(net(Ht), yht))
        if v < best - 1e-5:
            best, state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10:
                break
    net.load_state_dict(state); net.eval()
    return net.cpu()                                     # predict on the CPU


def weights(net):
    """Weights and biases of the linear layers of a trained network, in order, as float32 numpy arrays."""
    sd = net.state_dict()
    idx = sorted({int(k.split(".")[0]) for k in sd})
    return [(sd[f"{i}.weight"].detach().cpu().numpy().astype(np.float32), sd[f"{i}.bias"].detach().cpu().numpy().astype(np.float32))
            for i in idx]


def forward(W, A):
    """Class probabilities of the network with linear layers W (ReLU between them, softmax at the end), in numpy."""
    h = np.asarray(A, dtype=np.float32)
    for w, b in W[:-1]:
        h = np.maximum(h @ w.T + b, 0)
    z = h @ W[-1][0].T + W[-1][1]
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _main(src, dst):
    d = np.load(src)
    net = train(d["A"], d["ya"], d["H"], d["yh"], int(d["K"]), int(d["seed"]), int(d["threads"]))
    W = weights(net)
    np.savez(dst, n=len(W), **{f"w{i}": w for i, (w, _) in enumerate(W)}, **{f"b{i}": b for i, (_, b) in enumerate(W)})


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
