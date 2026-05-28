import torch

def area_resample_ref(x, out_h, out_w):
    _, _, in_h, in_w = x.shape
    def make_w(in_s, out_s, device, dtype):
        e_in  = torch.linspace(0, 1, in_s+1,  device=device, dtype=dtype)
        e_out = torch.linspace(0, 1, out_s+1, device=device, dtype=dtype)
        lo = torch.maximum(e_out[:-1,None], e_in[None,:-1])
        hi = torch.minimum(e_out[1:, None], e_in[None,1:])
        W = (hi-lo).clamp(min=0)
        return W / W.sum(1, keepdim=True)
    Wr = make_w(in_h, out_h, x.device, x.dtype)
    Wc = make_w(in_w, out_w, x.device, x.dtype)
    return torch.einsum('pi,bcij,qj->bcpq', Wr, x, Wc)

def area_resample_nn_correction(x, out_h, out_w):
    B, C, in_h, in_w = x.shape

    # --- NN base: each output pixel gets its containing input pixel ---
    p_idx = (torch.arange(out_h, device=x.device) * in_h // out_h).clamp(0, in_h-1)
    q_idx = (torch.arange(out_w, device=x.device) * in_w // out_w).clamp(0, in_w-1)
    out = x[:, :, p_idx, :][:, :, :, q_idx].clone()

    # --- find which output rows/cols straddle an input boundary ---
    def find_corrections(in_s, out_s):
        corr = {}
        for i in range(1, in_s):
            num = (i * out_s) % in_s   # 0 means perfectly aligned → no correction
            if num:
                corr[(i * out_s) // in_s] = (i, num / in_s)
        return corr

    h_corr = find_corrections(in_h, out_h)  # {output_row: (input_boundary, alpha)}
    v_corr = find_corrections(in_w, out_w)  # {output_col: (input_boundary, beta)}

    device, dtype = x.device, x.dtype

    def pack(corr):
        if not corr:
            return None
        ps = torch.tensor(list(corr.keys()),          device=device)
        bs = torch.tensor([v[0] for v in corr.values()], device=device)
        a  = torch.tensor([v[1] for v in corr.values()], device=device, dtype=dtype)
        return ps, bs, a

    h_pack = pack(h_corr)
    v_pack = pack(v_corr)

    # --- H corrections: rows straddling a horizontal input boundary ---
    if h_pack is not None:
        ps, bs, a = h_pack                                  # (Nh,)
        above = x[:, :, bs-1, :][:, :, :, q_idx]            # (B,C,Nh,out_w)
        below = x[:, :, bs,   :][:, :, :, q_idx]
        out[:, :, ps, :] = a[:, None] * above + (1-a)[:, None] * below

    # --- V corrections: cols straddling a vertical input boundary ---
    if v_pack is not None:
        qs, js, b = v_pack                                  # (Nv,)
        left  = x[:, :, :, js-1][:, :, p_idx, :]            # (B,C,out_h,Nv)
        right = x[:, :, :, js  ][:, :, p_idx, :]
        out[:, :, :, qs] = b * left + (1-b) * right

    # --- 2x2 corrections: pixels at the crossing of both boundaries ---
    if h_pack is not None and v_pack is not None:
        ps, bs, a = h_pack
        qs, js, b = v_pack
        # broadcast to (Nh,Nv)
        I0, J0 = bs[:, None] - 1, js[None, :] - 1
        I1, J1 = bs[:, None],     js[None, :]
        A, Bm = a[:, None], b[None, :]
        block = ( A     * Bm     * x[:, :, I0, J0]
                + A     * (1-Bm) * x[:, :, I0, J1]
                + (1-A) * Bm     * x[:, :, I1, J0]
                + (1-A) * (1-Bm) * x[:, :, I1, J1])         # (B,C,Nh,Nv)
        out[:, :, ps[:, None], qs[None, :]] = block

    return out


# --- tests ---
def check(name, x, oh, ow):
    ref = area_resample_ref(x, oh, ow)
    out = area_resample_nn_correction(x, oh, ow)
    ok = torch.allclose(ref, out, atol=1e-6)
    print(f"{name}: {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  ref:", ref)
        print("  got:", out)

check("center pixel 3x3->2x2", torch.tensor([[[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]]]), 2, 2)
check("checkerboard 2x2->3x3", torch.tensor([[[[1.,-1.],[-1.,1.]]]]), 3, 3)
check("1x1->3x3",              torch.tensor([[[[5.]]]]), 3, 3)
check("random 4x4->7x7",      torch.rand(2,3,4,4), 7, 7)
check("random 3x5->8x9",      torch.rand(1,1,3,5), 8, 9)