import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from scipy.ndimage import label

def spatial_entropy(attn_map_2d: torch.Tensor, threshold: float) -> Dict:
    # attn_map_2d: [P, P]
    S = attn_map_2d
    mean_val = torch.mean(S)
    B = torch.relu(S - mean_val*2)
    B_np = B.detach().cpu().to(torch.float32).numpy()
    binary = (B_np > threshold).astype(np.int32)

    labeled, num = label(binary, structure=np.ones((3, 3)))

    total = float(B.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}

    # Probability mass per component
    probs = []
    for i in range(1, num + 1):
        comp_sum = B_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}

def remove_singletons(mask_bool):
    structure = np.ones((3,3), dtype=bool)
    lab, _ = label(mask_bool.astype(bool), structure=structure)
    counts = np.bincount(lab.ravel())

    keep = np.zeros_like(counts, dtype=bool)
    keep[1:] = counts[1:] >= 2                # chỉ giữ các component size >= 2

    return keep[lab]

def get_kept_lh(attentions):
    sums = []
    for l in range(32):
        for h in range(32):
            s = float(attentions[l,h,:].sum())
            sums.append(s)

    thr_val = 0.49
    results: List[Dict] = []
    idx = 0
    for l in range(32):
        for h in range(32):
            s = sums[idx]
            idx += 1
            if s < thr_val:
                se = float("inf")
                bottom_row_focus = False
                n_comp = 0
            else:
                a2d = attentions[l, h, :].reshape(24, 24)
                se_res = spatial_entropy(a2d, 0.001)
                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > 0.05).any())
                se = float(se_res["spatial_entropy"])    # lower is better
                labeled = se_res["labeled_array"]
                n_comp = int(se_res["num_components"])
            results.append({
                "layer": l,
                "head": h,
                "attn_sum": s,
                "spatial_entropy": se,
                "bottom_row_focus": bottom_row_focus,
                "num_components": n_comp,
            })
    kept = [r for r in results if np.isfinite(r["spatial_entropy"]) and r["attn_sum"] >= thr_val and not r["bottom_row_focus"] and r["layer"] > 1]
    if len(kept) < 1:
        # fallback: take top by sum if too few
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: 1]

    kept.sort(key=lambda x: x["spatial_entropy"])
    return kept

SHELL_NOUNS = {
    "variety","kind","type","sort","form","category","class","genre","subtype","subset",
    "group","set","series","sequence","suite","lineup","selection","array","collection",
    "assortment","mix","blend","combination","mixture","package","bundle","batch",
    "bunch","cluster","stack","pile","heap","portfolio","inventory","list","range",
    "spectrum","continuum","aggregation","pool","bucket"
}

GENERIC_BUCKETS = {
    "entity","entities","object","objects","thing","things","item","items","unit","units",
    "component","components","element","elements","material","materials","content","contents",
    "product","products","article","articles","asset","assets","resource","resources","ingredient","ingredients",
    "stuff","substance","substances","artifact","artifacts","entry","entries","record","records"
}

MEASURE_NOUNS = {
    "amount","number","quantity","volume","mass","weight","size","degree","level","rate",
    "proportion","percentage","share","ratio","count","total","sum","average","mean","median",
    "portion","part","piece","section","segment","subset","member","instance","sample",
    "example","case","occurrence","pair","couple","trio","dozen","hundred","thousand","million"
}

IMAGE_DESCRIPTION_NOUNS = {
    "image", "photo", "picture", "scene", "view", "frame", "snapshot", "visual", "portrait", 
    "scene", "landscape", "depiction", "atmosphere", "illustration", "rendering", "capture"
}

DIRECTIONAL_NOUNS = {
    "top","bottom","middle","center","left","right","side","corner","edge","border","margin",
    "foreground","background","midground",
    "front","back","rear","frontside","backside","surface",
    "north","south","east","west","northeast","northwest","southeast","southwest"
}

OUTLIER_NOUNS = SHELL_NOUNS.union(GENERIC_BUCKETS).union(MEASURE_NOUNS).union(IMAGE_DESCRIPTION_NOUNS).union(DIRECTIONAL_NOUNS)