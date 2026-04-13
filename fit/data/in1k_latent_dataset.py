import os
import os.path as osp
import math
import torch
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
from einops import rearrange



class IN1kLatentDataset(Dataset):
    def __init__(self, root_dir, target_len=256, random='random'):
        super().__init__()
        self.RandomHorizontalFlipProb = 0.5
        self.root_dir = root_dir
        self.target_len = target_len
        self.random = random
        self.files = []
        files_1 = os.listdir(osp.join(root_dir, f'from_16_to_{target_len}'))
        files_2 = os.listdir(osp.join(root_dir, f'greater_than_{target_len}_resize'))
        files_3 = os.listdir(osp.join(root_dir, f'greater_than_{target_len}_crop'))
        files_23 = list(set(files_2) - set(files_3))    # files_3 in files_2
        self.files.extend([
            [osp.join(root_dir, f'from_16_to_{target_len}', file)] for file in files_1
        ])
        self.files.extend([
            [osp.join(root_dir, f'greater_than_{target_len}_resize', file)] for file in files_23
        ])
        self.files.extend([
            [
                osp.join(root_dir, f'greater_than_{target_len}_resize', file), 
                osp.join(root_dir, f'greater_than_{target_len}_crop', file)
            ] for file in files_3
        ])
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.random == 'random':
            path = random.choice(self.files[idx])
        elif self.random == 'resize':
            path = self.files[idx][0]  # only resize
        elif self.random == 'crop':
            path = self.files[idx][-1]  # only crop
        data = load_file(path)
        dtype = data['feature'].dtype
        
        feature = torch.zeros((self.target_len, 16), dtype=dtype)
        grid = torch.zeros((2, self.target_len), dtype=dtype)
        mask = torch.zeros((self.target_len), dtype=torch.uint8)
        size = torch.zeros(2, dtype=torch.int32)
        
        
        seq_len = data['grid'].shape[-1]
        if torch.rand(1) < self.RandomHorizontalFlipProb:
            feature[0: seq_len] = rearrange(data['feature'][0], 'h w c -> (h w) c')
        else:
            feature[0: seq_len] = rearrange(data['feature'][1], 'h w c -> (h w) c')
        grid[:, 0: seq_len] = data['grid']
        mask[0: seq_len] = 1
        size = data['size'][None, :]
        label = data['label']
        return dict(feature=feature, grid=grid, mask=mask, label=label, size=size)
        
       

def packed_collate_fn(samples, max_tokens: int = 512, pad_to_multiple: int = 128):
    """
    Greedy token-threshold collate function for packed / varlen attention training.

    Groups samples greedily: keeps adding images to the current sequence until
    the next image would push the total token count over `max_tokens`, then
    closes the group. This produces sequences with a roughly uniform number of
    valid tokens regardless of individual image size — much better for GPU
    utilisation than a fixed number of images per sequence.

    Args:
        samples: flat list of dicts from IN1kLatentDataset.__getitem__, each with
            keys feature (target_len, 16), grid (2, target_len), mask (target_len,),
            label (), size (1, 2).
        max_tokens: maximum total valid tokens per packed sequence.
        pad_to_multiple: pad N_total up to the next multiple of this value (must
            match FlexAttention's block size, typically 128).

    Returns a batched dict with:
        feature  (B, N_total, 16)
        grid     (B, 2, N_total)
        mask     (B, N_total)         — 1 for valid tokens, 0 for padding
        doc_ids  (B, N_total)         — image index within sequence, -1 for padding
        label    (B, max_n_pack)      — class labels, -1 for unused slots
        size     (B, max_n_pack, 2)   — (h, w) per image, 0 for unused slots
        n_pack   (B,)                 — number of images actually packed per element
    """
    def _seq_len(s):
        return int(s['mask'].sum())

    # ------------------------------------------------------------------ #
    # 1. Greedy grouping                                                   #
    # ------------------------------------------------------------------ #
    groups = []
    current_group = []
    current_len = 0
    for s in samples:
        slen = _seq_len(s)
        if current_group and current_len + slen > max_tokens:
            groups.append(current_group)
            current_group = [s]
            current_len = slen
        else:
            current_group.append(s)
            current_len += slen
    if current_group:
        groups.append(current_group)

    B = len(groups)
    max_n_pack = max(len(g) for g in groups)
    dtype_feat = samples[0]['feature'].dtype
    dtype_grid = samples[0]['grid'].dtype

    # ------------------------------------------------------------------ #
    # 2. Determine shared N_total (pad to multiple)                        #
    # ------------------------------------------------------------------ #
    raw_lens = [sum(_seq_len(s) for s in g) for g in groups]
    max_raw = max(raw_lens)
    N_total = math.ceil(max_raw / pad_to_multiple) * pad_to_multiple

    # ------------------------------------------------------------------ #
    # 3. Build per-group tensors and stack                                 #
    # ------------------------------------------------------------------ #
    feat_batch   = torch.zeros(B, N_total, 16, dtype=dtype_feat)
    grid_batch   = torch.zeros(B, 2, N_total, dtype=dtype_grid)
    mask_batch   = torch.zeros(B, N_total, dtype=torch.uint8)
    doc_batch    = torch.full((B, N_total), -1, dtype=torch.int32)
    label_batch  = torch.full((B, max_n_pack), -1, dtype=torch.int64)
    size_batch   = torch.zeros(B, max_n_pack, 2, dtype=torch.int32)
    n_pack_batch = torch.zeros(B, dtype=torch.int32)

    for b, group in enumerate(groups):
        offset = 0
        for img_idx, s in enumerate(group):
            slen = _seq_len(s)
            feat_batch[b, offset:offset + slen]     = s['feature'][:slen]
            grid_batch[b, :, offset:offset + slen]  = s['grid'][:, :slen]
            mask_batch[b, offset:offset + slen]     = 1
            doc_batch[b, offset:offset + slen]      = img_idx
            label_batch[b, img_idx]                 = s['label']
            size_batch[b, img_idx]                  = s['size'].squeeze(0)
            offset += slen
        n_pack_batch[b] = len(group)

    return dict(
        feature=feat_batch,
        grid=grid_batch,
        mask=mask_batch,
        doc_ids=doc_batch,
        label=label_batch,
        size=size_batch,
        n_pack=n_pack_batch,
    )


# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60

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
            self.train_config.data_path, self.train_config.target_len, self.train_config.random
        )
        
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, global_batch_size, max_steps, resume_step, seed=42,
                         packed=False, max_tokens=512, pad_to_multiple=128):
        """Build the training DataLoader.

        Args:
            packed: if True, use greedy token-threshold packing via packed_collate_fn
                instead of the default padding-based batching.
            max_tokens: maximum valid tokens per packed sequence (only used when packed=True).
            pad_to_multiple: pad N_total to this multiple for stable FlexAttention compilation
                (only used when packed=True).
        """
        sampler = get_train_sampler(
            self.train_dataset, global_batch_size, max_steps, resume_step, seed
        )
        collate_fn = None
        if packed:
            collate_fn = partial(
                packed_collate_fn,
                max_tokens=max_tokens,
                pad_to_multiple=pad_to_multiple,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
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
