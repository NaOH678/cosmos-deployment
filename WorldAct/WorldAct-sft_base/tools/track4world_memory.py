"""Diagnostic-only CPU storage for retained DA3 intermediate features.

No source files in Track4World are changed; preserve full attention context,
weights, dtype and feature normalization. Intended for eval, export_feat_layers=[].
"""

import inspect
import textwrap
import types

import torch


def offload_retained_features(backbone):
    original = backbone._get_intermediate_layers_not_chunked
    source = textwrap.dedent(inspect.getsource(original))
    needle = "output.append((out_x[:, :, 0], out_x))"
    assert source.count(needle) == 1
    source = source.replace(needle, "output.append((out_x[:, :, 0].cpu(), out_x.cpu()))")
    namespace = dict(original.__func__.__globals__)
    exec(compile(source, "<diagnostic_feature_offload>", "exec"), namespace)
    backbone._get_intermediate_layers_not_chunked = types.MethodType(namespace[original.__name__], backbone)

    def intermediate(self, x, n=1, export_feat_layers=None, **kwargs):
        if export_feat_layers:
            raise ValueError("Diagnostic offload only supports no auxiliary exported features")
        outputs, _ = self._get_intermediate_layers_not_chunked(x, n, export_feat_layers=[], **kwargs)
        normalized, cameras = [], []
        for camera, feature in outputs:
            feature = feature.to(x.device)
            cameras.append(camera.to(x.device))
            if feature.shape[-1] == self.embed_dim:
                feature = self.norm(feature)
            elif feature.shape[-1] == self.embed_dim * 2:
                feature = torch.cat([feature[..., : self.embed_dim], self.norm(feature[..., self.embed_dim :])], dim=-1)
            else:
                raise ValueError("Unexpected feature dimension")
            normalized.append(feature[..., 1 + self.num_register_tokens :, :])
        return tuple(zip(normalized, cameras)), []

    backbone.get_intermediate_layers = types.MethodType(intermediate, backbone)
