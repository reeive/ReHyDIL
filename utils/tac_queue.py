from typing import List, Sequence, Optional, Tuple
import random
import torch
import torch.nn as nn


def _to_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().tolist()
    return [x]


def tversky_sim(p: torch.Tensor, q: torch.Tensor, alpha: float = 0.7, beta: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample per-channel Tversky similarity.
    Args:
        p, q: [N, C, H, W] probabilities in [0,1].
    Returns:
        sim: [N, C]
    """
    assert p.shape == q.shape, f"Shape mismatch: {p.shape} vs {q.shape}"
    tp = (p * q).sum(dim=(2, 3))
    fp = ((1.0 - q) * p).sum(dim=(2, 3))
    fn = ((1.0 - p) * q).sum(dim=(2, 3))
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


class PredictionQueue:
    """Simple bounded queue for teacher predictions and metadata.

    Stores tensors on CPU (float16 by default) to save GPU memory;
    tensors are moved to the target device only when sampled.
    """
    def __init__(self, max_size: int, channels: int, height: int, width: int, dtype=torch.float16, device: str = "cpu"):
        self.max_size = int(max_size)
        self.channels = channels
        self.height = height
        self.width = width
        self.dtype = dtype
        self.device = device

        self._probs = torch.empty((0, channels, height, width), dtype=dtype, device=device)
        self._patients: List[str] = []
        self._modalities: List[str] = []

    @property
    def size(self) -> int:
        return self._probs.shape[0]

    def enqueue(self, probs: torch.Tensor, patient_ids: Sequence, modality_ids: Sequence):
        """Append a batch.
        probs: [N, C, H, W] on any device (will be moved to self.device/dtype)
        patient_ids/modality_ids: len N (list/tuple/tensor)
        """
        if probs.numel() == 0:
            return
        assert probs.dim() == 4 and probs.shape[1:] == (self.channels, self.height, self.width), \
            f"probs shape {probs.shape} != (N,{self.channels},{self.height},{self.width})"
        pt = _to_list(patient_ids)
        md = _to_list(modality_ids)
        assert len(pt) == probs.shape[0] and len(md) == probs.shape[0], "metadata length must match batch"

        probs_cpu = probs.detach().to(self.device, dtype=self.dtype)
        self._probs = torch.cat([self._probs, probs_cpu], dim=0)
        self._patients.extend([str(x) for x in pt])
        self._modalities.extend([str(x) for x in md])

        # Truncate from the front if exceeding max_size
        if self.size > self.max_size:
            overflow = self.size - self.max_size
            self._probs = self._probs[overflow:].contiguous()
            self._patients = self._patients[overflow:]
            self._modalities = self._modalities[overflow:]

    def sample_for_batch(self,
                         current_patient_ids: Sequence,
                         current_modality_ids: Sequence,
                         num: int) -> Tuple[torch.Tensor, List[int]]:
        """Sample up to `num` items so that sampled items differ in patient & modality from current batch.
        Returns:
            probs: [S, C, H, W] (CPU float16) — caller can move to GPU as needed
            idxs:  list of indices sampled from the queue
        """
        Nq = self.size
        if Nq == 0:
            return torch.empty(0, self.channels, self.height, self.width, dtype=self.dtype, device=self.device), []

        cur_pat = [str(x) for x in _to_list(current_patient_ids)]
        cur_mod = [str(x) for x in _to_list(current_modality_ids)]

        # Build a candidate index pool that does not share patient with current batch (strictest constraint)
        bad_patients = set(cur_pat)
        cand = [i for i, p in enumerate(self._patients) if p not in bad_patients]
        if not cand:
            cand = list(range(Nq))  # fallback

        S = min(int(num), len(cand))
        random.shuffle(cand)
        idx = cand[:S]
        return self._probs[idx], idx


class TACWithQueues(nn.Module):
    """Tversky-InfoNCE using teacher queue.

    Positives: same channel between current batch prediction and a sampled teacher prediction.
    Negatives: channel-rotated pairs (mismatched region) + optional random mismatches.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 1.5, tau: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

    def forward(self,
                cur_probs: torch.Tensor,           # [B, C, H, W] on GPU
                cur_patient_ids: Sequence,
                cur_modality_ids: Sequence,
                teacher_queue: PredictionQueue) -> torch.Tensor:
        B, C, H, W = cur_probs.shape
        # Sample S from queue (CPU)
        q_probs_cpu, idx = teacher_queue.sample_for_batch(cur_patient_ids, cur_modality_ids, num=B)
        S = q_probs_cpu.shape[0]
        if S == 0:
            return cur_probs.new_tensor(0.0)

        # Move sampled teacher probs to cur device as float32 for numerics
        device = cur_probs.device
        q_probs = q_probs_cpu.to(device=device, dtype=cur_probs.dtype, non_blocking=True)

        # Choose S items from the current batch to align with queue samples
        if S < B:
            sel = torch.randperm(B, device=device)[:S]
            p_probs = cur_probs[sel]
        else:
            p_probs = cur_probs[:S]

        # Tversky sims (pos: same channel)
        sim_pos = tversky_sim(p_probs, q_probs, alpha=self.alpha, beta=self.beta)  # [S, C]

        # Negatives via channel rotation (mismatch region) + random shuffle
        q_probs_neg = torch.roll(q_probs, shifts=1, dims=1)
        sim_neg1 = tversky_sim(p_probs, q_probs_neg, alpha=self.alpha, beta=self.beta)  # [S, C]

        # Optional extra negatives: shuffle along batch
        if S > 1:
            perm = torch.randperm(S, device=device)
            q_probs_neg2 = q_probs[perm]
            sim_neg2 = tversky_sim(p_probs, q_probs_neg2, alpha=self.alpha, beta=self.beta)  # [S, C]
            sim_neg = torch.cat([sim_neg1, sim_neg2], dim=1)  # [S, 2C]
        else:
            sim_neg = sim_neg1  # [1, C]

        # InfoNCE across channel dimension: aggregate by mean for stability
        logit_pos = sim_pos / self.tau  # [S, C]
        logit_neg = sim_neg / self.tau  # [S, K]

        # Numerically stable log-softmax style objective
        max_pos = logit_pos.max(dim=1, keepdim=True).values
        max_neg = logit_neg.max(dim=1, keepdim=True).values
        Z_pos = (logit_pos - max_pos).exp().mean(dim=1)  # mean over channels
        Z_neg = (logit_neg - max_neg).exp().mean(dim=1)
        loss = -torch.log(Z_pos / (Z_pos + Z_neg + 1e-12)).mean()
        return loss


def build_teacher_queue(prev_model: nn.Module,
                        images: torch.Tensor,
                        patient_ids: Sequence,
                        modality_ids: Sequence,
                        max_size: int,
                        batch_size: int = 64,
                        device: str = "cuda",
                        out_device: str = "cpu",
                        dtype = torch.float16) -> PredictionQueue:
    """Run prev_model over replay images to populate a teacher queue.
    images: [N, C, H, W] float tensor (can be CPU/GPU)
    """
    N, C, H, W = images.shape
    q = PredictionQueue(max_size=max_size, channels=C, height=H, width=W, dtype=dtype, device=out_device)
    prev_model.eval().to(device)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            sl = slice(i, min(i + batch_size, N))
            x = images[sl].to(device, non_blocking=True)
            y = torch.sigmoid(prev_model(x)).detach()  # [n, C, H, W]
            q.enqueue(y, patient_ids=patient_ids[sl], modality_ids=modality_ids[sl])
    return q
