import os
import os.path as osp
import math
import torch
import torch.nn.functional as F
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset, BatchSampler
from safetensors.torch import load_file
from einops import rearrange



class IN1kLatentDataset(Dataset):
    def __init__(self, root_dir, target_len=256, random='random',
                 resize_range=None, return_fullres=False):
        super().__init__()
        self.RandomHorizontalFlipProb = 0.5
        self.root_dir = root_dir
        self.target_len = target_len
        self.random = random
        self.resize_range = resize_range      # (min_grid, max_grid) or None
        self.return_fullres = return_fullres  # if True, also return full-res feature
        self.files = []
        dir_1 = osp.join(root_dir, f'from_16_to_{target_len}')
        dir_2 = osp.join(root_dir, f'greater_than_{target_len}_resize')
        dir_3 = osp.join(root_dir, f'greater_than_{target_len}_crop')
        files_1 = os.listdir(dir_1) if osp.isdir(dir_1) else []
        files_2 = os.listdir(dir_2) if osp.isdir(dir_2) else []
        files_3 = os.listdir(dir_3) if osp.isdir(dir_3) else []
        files_23 = list(set(files_2) - set(files_3))    # files_3 in files_2
        self.files.extend([
            [osp.join(dir_1, file)] for file in files_1
        ])
        self.files.extend([
            [osp.join(dir_2, file)] for file in files_23
        ])
        self.files.extend([
            [
                osp.join(dir_2, file),
                osp.join(dir_3, file)
            ] for file in files_3
        ])
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, override_g = idx
        else:
            override_g = None
        if self.random == 'random':
            path = random.choice(self.files[idx])
        elif self.random == 'resize':
            path = self.files[idx][0]  # only resize
        elif self.random == 'crop':
            path = self.files[idx][-1]  # only crop
        data = load_file(path)
        dtype = data['feature'].dtype
        p = 2   # patch_size (matches model)
        C = 4   # VAE latent channels

        # Pick flip variant
        if torch.rand(1) < self.RandomHorizontalFlipProb:
            feat_hw = data['feature'][0]   # (H_g, W_g, 16)
        else:
            feat_hw = data['feature'][1]

        H_g, W_g = feat_hw.shape[0], feat_hw.shape[1]

        # Stash full-res before any resize (only when needed)
        feat_hw_fullres = feat_hw if self.return_fullres else None

        # Optional random bilinear resize to a smaller even grid size
        if self.resize_range is not None:
            min_g, max_g = self.resize_range
            valid = list(range(min_g, max_g + 1, 2))
            new_g = override_g if override_g is not None else random.choice(valid)
            if new_g != H_g:
                # Unpatchify: (H_g, W_g, p²C) → (H_g*p, W_g*p, C)
                feat_latent = rearrange(feat_hw.float(),
                                        'h w (p1 p2 c) -> (h p1) (w p2) c',
                                        p1=p, p2=p, c=C)
                # Bilinear resize at latent level
                feat_t = feat_latent.permute(2, 0, 1).unsqueeze(0)  # (1, C, H_l, W_l)
                feat_t = F.interpolate(feat_t, size=(new_g * p, new_g * p),
                                       mode='bilinear', align_corners=True)
                feat_latent = feat_t.squeeze(0).permute(1, 2, 0).to(dtype)
                # Re-patchify: (H_l', W_l', C) → (H_g', W_g', p²C)
                feat_hw = rearrange(feat_latent,
                                    '(h p1) (w p2) c -> h w (p1 p2 c)',
                                    p1=p, p2=p)
                H_g, W_g = new_g, new_g

        seq_len = H_g * W_g

        # Build padded output tensors
        feature = torch.zeros((self.target_len, 16), dtype=dtype)
        grid    = torch.zeros((2, self.target_len), dtype=dtype)
        mask    = torch.zeros((self.target_len,), dtype=torch.uint8)

        feature[:seq_len] = feat_hw.reshape(seq_len, 16)

        # Recompute integer grid for the current resolution
        hs = torch.arange(H_g, dtype=dtype)
        ws = torch.arange(W_g, dtype=dtype)
        gh, gw = torch.meshgrid(hs, ws, indexing='ij')
        grid[:, :seq_len] = torch.stack([gh.reshape(-1), gw.reshape(-1)])

        mask[:seq_len] = 1
        size  = torch.tensor([[H_g, W_g]], dtype=torch.int32)
        label = data['label']

        result = dict(feature=feature, grid=grid, mask=mask, label=label, size=size)

        if self.return_fullres:
            seq_fr = int(data['grid'].shape[-1])
            feat_fr = torch.zeros((self.target_len, 16), dtype=dtype)
            mask_fr = torch.zeros((self.target_len,), dtype=torch.uint8)
            feat_fr[:seq_fr] = feat_hw_fullres.reshape(seq_fr, 16)
            mask_fr[:seq_fr] = 1
            H_fr = int(data['size'][0])
            W_fr = int(data['size'][1])
            result['feature_fullres'] = feat_fr
            result['mask_fullres']    = mask_fr
            result['size_fullres']    = torch.tensor([[H_fr, W_fr]], dtype=torch.int32)

        return result
        
       

class TokenBudgetBatchSampler(BatchSampler):
    """Yield index batches such that the total valid tokens per batch stays
    within *max_tokens*.

    Token counts are pre-computed without touching the dataset files: each
    sample's grid size is drawn from the same uniform distribution used in
    IN1kLatentDataset.__getitem__ (resize_range → H_g*W_g), seeded
    deterministically so the lengths are reproducible across restarts.
    The pre-sampled grid size is encoded into the yielded indices as
    (dataset_idx, grid_size) tuples so that __getitem__ uses the same draw
    instead of re-sampling independently.

    Args:
        sampler:        A flat sequence of dataset indices (pre-shuffled by
                        get_train_sampler).  Consumed in order.
        resize_range:   (min_grid, max_grid) passed to the dataset.  If None,
                        every sample is assumed to be target_len tokens.
        target_len:     Per-sample capacity used when resize_range is None.
        max_tokens:     Token budget per batch.
        pad_to_multiple: Ignored here (handled by collate); kept for symmetry.
        seed:           RNG seed for the length pre-computation.
    """

    def __init__(self, sampler, resize_range=None, target_len=256,
                 max_tokens=1024, pad_to_multiple=128, seed=42):
        # Don't call super().__init__() — it expects (sampler, batch_size, drop_last)
        # and we don't have a fixed batch_size.
        self.sampler = sampler          # list of indices
        self.max_tokens = max_tokens
        self.pad_to_multiple = pad_to_multiple

        # Pre-compute a grid size and seq_len for every position in the sampler list.
        # The grid size is also encoded into the yielded index tuples so that
        # __getitem__ uses the same draw rather than re-sampling independently.
        if resize_range is not None:
            min_g, max_g = resize_range
            valid = list(range(min_g, max_g + 1, 2))
            rng = random.Random(seed)
            self._grid_sizes = [rng.choice(valid) for _ in sampler]
            self._lengths = [g ** 2 for g in self._grid_sizes]
        else:
            self._grid_sizes = [None] * len(sampler)
            self._lengths = [target_len] * len(sampler)

    def __iter__(self):
        batch, total = [], 0
        for idx, grid_size, length in zip(self.sampler, self._grid_sizes, self._lengths):
            if batch and total + length > self.max_tokens:
                yield batch
                batch, total = [], 0
            batch.append((idx, grid_size))
            total += length
        if batch:
            yield batch

    def __len__(self):
        # Approximate — actual number of batches depends on sampled lengths.
        avg = sum(self._lengths) / max(len(self._lengths), 1)
        return math.ceil(len(self.sampler) * avg / self.max_tokens)


def packed_collate_fn(samples, pad_to_multiple: int = 128):
    """
    Collate a pre-grouped list of samples into a single packed sequence.

    Grouping is handled upstream by TokenBudgetBatchSampler, so all samples
    here belong to one sequence.  We just concatenate, assign doc_ids, and
    pad N_total to the nearest multiple of pad_to_multiple.

    Args:
        samples: list of dicts from IN1kLatentDataset.__getitem__, each with
            keys feature (target_len, 16), grid (2, target_len), mask (target_len,),
            label (), size (1, 2).
        pad_to_multiple: pad N_total to this multiple for FlexAttention block size.

    Returns a batched dict with shape (1, N_total, ...) where B=1:
        feature  (1, N_total, 16)
        grid     (1, 2, N_total)
        mask     (1, N_total)         — 1 for valid tokens, 0 for padding
        doc_ids  (1, N_total)         — image index within sequence, -1 for padding
        label    (1, n_pack)          — class labels
        size     (1, n_pack, 2)       — (h, w) per image
        n_pack   (1,)                 — number of images packed
    Also propagates full-res fields when present (return_fullres=True):
        feature_fullres  (1, N_total_fr, 16)
        mask_fullres     (1, N_total_fr)
        doc_ids_fr       (1, N_total_fr)  — image index for full-res tokens, -1 for padding
        size_fullres     (1, n_pack, 2)   — (H_fr, W_fr) per image
    """
    def _seq_len(s):
        return int(s['mask'].sum())

    n_pack = len(samples)
    dtype_feat = samples[0]['feature'].dtype
    dtype_grid = samples[0]['grid'].dtype

    raw_len = sum(_seq_len(s) for s in samples)
    N_total = math.ceil(raw_len / pad_to_multiple) * pad_to_multiple

    feat_batch  = torch.zeros(1, N_total, 16, dtype=dtype_feat)
    grid_batch  = torch.zeros(1, 2, N_total, dtype=dtype_grid)
    mask_batch  = torch.zeros(1, N_total, dtype=torch.uint8)
    doc_batch   = torch.full((1, N_total), -1, dtype=torch.int32)
    label_batch = torch.full((1, n_pack), -1, dtype=torch.int64)
    size_batch  = torch.zeros(1, n_pack, 2, dtype=torch.int32)

    offset = 0
    for img_idx, s in enumerate(samples):
        slen = _seq_len(s)
        feat_batch[0, offset:offset + slen]    = s['feature'][:slen]
        grid_batch[0, :, offset:offset + slen] = s['grid'][:, :slen]
        mask_batch[0, offset:offset + slen]    = 1
        doc_batch[0, offset:offset + slen]     = img_idx
        label_batch[0, img_idx]                = s['label']
        size_batch[0, img_idx]                 = s['size'].squeeze(0)
        offset += slen

    result = dict(
        feature=feat_batch,
        grid=grid_batch,
        mask=mask_batch,
        doc_ids=doc_batch,
        label=label_batch,
        size=size_batch,
        n_pack=torch.tensor([n_pack], dtype=torch.int32),
    )

    # Propagate full-res fields when the dataset was built with return_fullres=True.
    if 'feature_fullres' in samples[0]:
        raw_len_fr = sum(int(s['mask_fullres'].sum()) for s in samples)
        N_total_fr = math.ceil(raw_len_fr / pad_to_multiple) * pad_to_multiple

        feat_fr_batch = torch.zeros(1, N_total_fr, 16, dtype=dtype_feat)
        mask_fr_batch = torch.zeros(1, N_total_fr, dtype=torch.uint8)
        doc_fr_batch  = torch.full((1, N_total_fr), -1, dtype=torch.int32)
        size_fr_batch = torch.zeros(1, n_pack, 2, dtype=torch.int32)

        offset_fr = 0
        for img_idx, s in enumerate(samples):
            slen_fr = int(s['mask_fullres'].sum())
            feat_fr_batch[0, offset_fr:offset_fr + slen_fr] = s['feature_fullres'][:slen_fr]
            mask_fr_batch[0, offset_fr:offset_fr + slen_fr] = 1
            doc_fr_batch[0, offset_fr:offset_fr + slen_fr]  = img_idx
            size_fr_batch[0, img_idx]                        = s['size_fullres'].squeeze(0)
            offset_fr += slen_fr

        result['feature_fullres'] = feat_fr_batch  # (1, N_total_fr, 16)
        result['mask_fullres']    = mask_fr_batch  # (1, N_total_fr)
        result['doc_ids_fr']      = doc_fr_batch   # (1, N_total_fr)
        result['size_fullres']    = size_fr_batch  # (1, n_pack, 2)

    return result


# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60
def get_train_sampler(dataset, global_batch_size, max_steps, resume_steps, seed):
    sample_indices = torch.empty([max_steps * global_batch_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[
            :sample_indices.size(0) - fill_ptr
        ]
        sample_indices[fill_ptr: fill_ptr + epoch_sample_indices.size(0)] = \
            epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_steps * global_batch_size : ].tolist()


   
class INLatentLoader():
    def __init__(self, train):
        super().__init__()

        self.train_config = train

        self.batch_size = self.train_config.loader.batch_size
        self.num_workers = self.train_config.loader.num_workers
        self.shuffle = self.train_config.loader.shuffle

        self.train_dataset = IN1kLatentDataset(
            self.train_config.data_path,
            self.train_config.target_len,
            self.train_config.random,
            resize_range=getattr(self.train_config, 'resize_range', None),
            return_fullres=getattr(self.train_config, 'return_fullres', False),
        )
        
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, global_batch_size, max_steps, resume_step, seed=42,
                         packed=False, max_tokens=512, pad_to_multiple=128):
        """Build the training DataLoader.

        Args:
            packed: if True, use TokenBudgetBatchSampler + packed_collate_fn.
                The sampler groups indices so that total valid tokens per batch
                stays within max_tokens; batch_size from config is ignored.
            max_tokens: token budget per packed batch (only used when packed=True).
            pad_to_multiple: pad N_total to this multiple for FlexAttention
                (only used when packed=True).
        """
        flat_sampler = get_train_sampler(
            self.train_dataset, global_batch_size, max_steps, resume_step, seed
        )
        if packed:
            resize_range = getattr(self.train_config, 'resize_range', None)
            batch_sampler = TokenBudgetBatchSampler(
                sampler=flat_sampler,
                resize_range=resize_range,
                target_len=self.train_config.target_len,
                max_tokens=max_tokens,
                pad_to_multiple=pad_to_multiple,
                seed=seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=partial(packed_collate_fn, pad_to_multiple=pad_to_multiple),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=flat_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
