from pathlib import Path
import difflib
B=Path(__file__).resolve().parent
p=B/'Track4World/track4world/nets/model.py'
src=Path('/mnt/afs/WorldAct-pointflow-native/Track4World_portable/Track4World/track4world/nets/model.py')
s=src.read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old))
 s=s.replace(old,new)
replace('import torch.version\n','import torch.version\nfrom track4world.nets.da3_chunk_geometry import (\n    target_in_source_units, align_pairs_in_source_units, restore_geometry,\n)\n')
# Preserve exactly the existing per-chunk DA3 focal convention. No calibration fitting.
replace('        # Iterate through chunks of frames along the time dimension\n', '''        # Reset call-local metadata; never reuse a previous video's final block.
        self._da3_frame_scales = None
        self._da3_frame_focals = None
        self._da3_frame_chunk_ids = None
        da3_scales, da3_focals, da3_chunk_ids = [], [], []
        if self.use_model == 'depthanythingv3' and B != 1:
            raise ValueError('DA3 EFEP metadata currently supports B=1 only')

        # Iterate through chunks of frames along the time dimension
''')
replace('            # Append reshaped outputs\n', '''            if self.use_model == 'depthanythingv3':
                count = images_chunk.shape[1]
                da3_scales.append(self._metric_scale.detach().expand(count).clone())
                da3_focals.append(self._da3_focal.detach().expand(count).clone())
                da3_chunk_ids.append(torch.full((count,), t // fmaps_chunk_size,
                                               device=pt_c.device, dtype=torch.long))

            # Append reshaped outputs
''')
replace('        # Final concatenation and flattening for downstream processing\n', '''        if da3_scales:
            self._da3_frame_scales = torch.cat(da3_scales)
            self._da3_frame_focals = torch.cat(da3_focals)
            self._da3_frame_chunk_ids = torch.cat(da3_chunk_ids)

        # Final concatenation and flattening for downstream processing
''')
replace('        pms = self.pairwise_concat(pms)\n', '''        pms = self.pairwise_concat(pms)
        if self.use_model == 'depthanythingv3':
            # 3D correlation and predicted displacement use the source scale.
            pms = target_in_source_units(pms, self._da3_frame_scales)
''')
# Limit output changes to infer_pair, leaving other inference entrypoints untouched.
pos=s.index('    def infer_pair(');prefix=s[:pos];tail=s[pos:]
def edit(old,new):
 global tail
 assert tail.count(old)==1,(old[:100],tail.count(old))
 tail=tail.replace(old,new)
edit('        if aligned_scene_flow:\n', '''        da3_scales = (self._da3_frame_scales
                      if self.use_model == 'depthanythingv3' else None)
        if da3_scales is not None and da3_scales.numel() != T:
            raise RuntimeError('DA3 frame metadata does not match output frames')
        if aligned_scene_flow and da3_scales is not None:
            flow3d, visconf_maps_e = align_pairs_in_source_units(
                flow2d_c, flow3d, points,
                visconf_maps_e.cuda().permute(0, 1, 3, 4, 2),
                da3_scales, self._da3_frame_chunk_ids,
                get_aligned_scene_flow_temporal,
                refine_confidence_with_geometric_consistency)
        elif aligned_scene_flow:
''')
edit('            if _ms is not None:\n', '''            if _ms is not None and da3_scales is not None:
                points, flow3d, world_points, camera_poses = restore_geometry(
                    points, flow3d, world_points, camera_poses, da3_scales)
            elif _ms is not None:
''')
edit('            if _da3_f is not None:\n', '''            if _da3_f is not None and da3_scales is not None:
                focal = self._da3_frame_focals[None].to(points)
                shift = torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
            elif _da3_f is not None:
''')
edit('''            intrinsics = utils3d.torch.intrinsics_from_focal_center(
                fx, fy, 0.5, 0.5
            ).repeat(1, points.shape[1], 1, 1)''','''            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
            if focal.ndim == 1:
                intrinsics = intrinsics.repeat(1, points.shape[1], 1, 1)''')
edit("                    intrinsics=intrinsics[:, :-1][..., None, :, :],", "                    intrinsics=(intrinsics[:, 1:] if da3_scales is not None else intrinsics[:, :-1])[..., None, :, :],")
# Attach explicit coordinate/normalization provenance to raw results.
edit('        return return_dict\n', '''        if da3_scales is not None:
            return_dict[0]['metric_scale'] = da3_scales
            return_dict[1]['metric_scale'] = da3_scales[:-1]
            return_dict[0]['frame_metric_scales'] = da3_scales
            return_dict[0]['frame_normalized_focals'] = self._da3_frame_focals
            return_dict[0]['frame_chunk_ids'] = self._da3_frame_chunk_ids
            return_dict[0]['world_coordinate_scope'] = 'backbone_chunk'
        return return_dict
''')
s=prefix+tail
p.write_text(s)
(B/'model.patch').write_text(''.join(difflib.unified_diff(src.read_text().splitlines(True),s.splitlines(True),fromfile='a/track4world/nets/model.py',tofile='b/track4world/nets/model.py')))
print('Isolated model patched:',p)
