"""Per-frame DA3 geometry metadata for single-video EFEP inference.

Network scene flow is expressed in the source frame's normalized units.
DA3 world poses remain local to each backbone chunk; this module does not
estimate a cross-chunk world-frame registration.
"""
import torch


def target_in_source_units(pair_pointmaps, scales):
    """[T-1,2,3,h,w] pointmaps, [T] positive normalization scales."""
    if pair_pointmaps.shape[:2] != (scales.numel()-1, 2):
        raise ValueError('Point pair count and frame scales do not match')
    out = pair_pointmaps.clone()
    out[:, 1] *= (scales[1:] / scales[:-1]).to(out)[:, None, None, None]
    return out


def align_pairs_in_source_units(flow2d, endpoints, points, confidence,
                                scales, chunk_ids, align_fn, confidence_fn):
    """Keep native within-chunk alignment; rescale the target at chunk edges.

    Process runs with one source scale so the original normalization and
    confidence sensitivity are unchanged inside each DA3 chunk.
    """
    if points.shape[0] != 1 or scales.shape != (points.shape[1],):
        raise ValueError('DA3 chunk alignment currently supports one video')
    aligned, refined = [], []
    starts = [0] + (torch.nonzero(chunk_ids[1:] != chunk_ids[:-1]).flatten()+1).tolist()
    for start, end in zip(starts, starts[1:]+[points.shape[1]]):
        stop = min(end, points.shape[1]-1)
        if start >= stop:
            continue
        local = points[:, start:stop+1] * (scales[start:stop+1]/scales[start]).to(points)[None,:,None,None,None]
        flow = align_fn(flow2d[:,start:stop], endpoints[:,start:stop], local,
                        confidence[:,start:stop], mode='align_dir')
        conf = confidence_fn(flow2d[:,start:stop], flow, local,
                             confidence[:,start:stop])
        aligned.append(flow)
        refined.append(conf)
    return torch.cat(aligned,dim=1), torch.cat(refined,dim=1)


def restore_geometry(points, endpoints, world_points, camera_poses, scales):
    """Restore per-frame geometry and source-normalized pair endpoints."""
    s = scales.to(points)
    points = points * s[None,:,None,None,None]
    endpoints = endpoints * s[:-1].to(endpoints)[None,:,None,None,None]
    world_points = world_points * s.to(world_points)[None,:,None,None,None]
    camera_poses = camera_poses.clone()
    camera_poses[..., :3, 3] *= s.to(camera_poses)[...,None]
    return points, endpoints, world_points, camera_poses
