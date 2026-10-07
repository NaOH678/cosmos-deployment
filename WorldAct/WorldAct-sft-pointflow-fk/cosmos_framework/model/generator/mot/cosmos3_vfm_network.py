# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from typing import List, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.data.generator.sequence_packing.natten import verify_natten_parameter_list
from cosmos_framework.model.generator.fk_sequence import attach_fk_tokens, fk_positions
from cosmos_framework.model.generator.mot.attention import SplitInfo, build_packed_sequence
from cosmos_framework.model.generator.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
    get_context_parallel_sharded_sequence,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder, has_noisy_tokens
from cosmos_framework.model.generator.utils.memory import MemoryState


class Cosmos3VFMNetworkConfig(PretrainedConfig):
    def __init__(
        self,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        vlm_config=None,
        latent_patch_size=2,
        latent_downsample_factor=8,
        latent_channel_size=16,
        max_latent_h=32,
        max_latent_w=32,
        max_latent_t=32,
        enable_fps_modulation=False,
        base_fps=24,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        timestep_scale=0.001,
        predict_text_tokens=False,
        joint_attn_implementation="two_way",
        action_dim=32,
        num_embodiment_domains=32,
        temporal_compression_factor_vision=4,
        temporal_compression_factor_action=1,
        natten_parameter_list=None,
        video_temporal_causal=False,
        # Sound generation parameters
        sound_dim: int | None = None,
        temporal_compression_factor_sound=1,
        sound_latent_fps: int = 25,
        enable_input_bias: bool = True,
        **kwargs,
    ):
        self.vision_gen = vision_gen
        self.sound_gen = sound_gen
        self.vlm_config = vlm_config
        self.latent_patch_size = latent_patch_size
        self.latent_downsample_factor = latent_downsample_factor
        self.latent_channel_size = latent_channel_size
        self.max_latent_h = max_latent_h
        self.max_latent_w = max_latent_w
        self.max_latent_t = max_latent_t
        self.enable_fps_modulation = enable_fps_modulation
        self.base_fps = base_fps
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.timestep_scale = timestep_scale
        self.predict_text_tokens = predict_text_tokens
        self.joint_attn_implementation = joint_attn_implementation
        self.temporal_compression_factor_vision = temporal_compression_factor_vision
        self.natten_parameter_list = natten_parameter_list
        self.video_temporal_causal = video_temporal_causal
        self.enable_input_bias = enable_input_bias

        # action related parameters
        self.action_gen = action_gen  # whether to generate action tokens
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        self.temporal_compression_factor_action = temporal_compression_factor_action
        if self.action_gen:
            assert self.vision_gen, (
                "Action generation requires visual generation! We do NOT support action only training!"
            )

        # sound related parameters
        self.sound_dim = sound_dim
        self.temporal_compression_factor_sound = temporal_compression_factor_sound
        self.sound_latent_fps = sound_latent_fps
        if self.sound_gen:
            assert self.vision_gen, (
                "Sound generation requires visual generation! We do NOT support sound only training!"
            )

        super().__init__(**kwargs)


class Cosmos3VFMNetwork(PreTrainedModel):
    config_class = Cosmos3VFMNetworkConfig
    base_model_prefix = "cosmos3"

    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(config)
        self.language_model = language_model

        text_config = config.vlm_config.text_config if hasattr(config.vlm_config, "text_config") else config.vlm_config
        self.hidden_size = text_config.hidden_size
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.num_hidden_layers = text_config.num_hidden_layers
        self.attention_io_layout = "sequence_sharded"
        self.predict_text_tokens = config.predict_text_tokens

        if config.natten_parameter_list is not None and config.joint_attn_implementation != "three_way":
            raise NotImplementedError(
                f"Sparsity is only supported with 'three_way' attention, but got {config.joint_attn_implementation=}, "
                "and 'natten_parameter_list' was not None."
            )
        self.natten_parameter_list = verify_natten_parameter_list(
            config.natten_parameter_list, num_layers=self.num_hidden_layers
        )

        if config.video_temporal_causal and config.joint_attn_implementation != "three_way":
            raise ValueError(
                f"video_temporal_causal=True requires joint_attn_implementation='three_way', "
                f"but got {config.joint_attn_implementation!r}."
            )
        self.video_temporal_causal = config.video_temporal_causal
        self.pad_for_cuda_graphs = False

        if config.vision_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.timestep_scale = config.timestep_scale
            self.latent_downsample = config.latent_downsample_factor * config.latent_patch_size
            self.max_latent_h = config.max_latent_h
            self.max_latent_w = config.max_latent_w
            self.max_latent_t = config.max_latent_t
            self.latent_channel = config.latent_channel_size
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            _input_bias = config.enable_input_bias
            self.time_embedder = TimestepEmbedder(self.hidden_size, bias=_input_bias)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size, bias=_input_bias)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

        if config.action_gen:
            self.action_dim = config.action_dim
            self.num_embodiment_domains = config.num_embodiment_domains
            self.action2llm = DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.llm2action = DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)

            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if config.sound_gen:
            self.sound_dim = config.sound_dim
            self.sound2llm = nn.Linear(config.sound_dim, self.hidden_size, bias=config.enable_input_bias)
            self.llm2sound = nn.Linear(self.hidden_size, config.sound_dim)
            self.sound_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        self.config = config
        self.parallel_dims = None
        self.pointflow_branch = None
        self.fk_branch = None

    def install_local_geometry_rope(
        self, *, depth_min=0.125, depth_max=4.0, rho0=1.0, rho_scale=1.0, implementation="partition"
    ):
        from cosmos_framework.model.generator.pointflow_attention import reference_attention_enabled
        from cosmos_framework.utils import log

        if (
            self.head_dim != 128
            or list(self.language_model.config.mrope_section) != [24, 20, 20]
            or self.config.joint_attn_implementation != "two_way"
            or self.video_temporal_causal
            or reference_attention_enabled()
        ):
            raise ValueError("Local geometry RoPE requires Edge 128/[24,20,20], two-way and no reference attention")
        if not (0 < depth_min <= depth_max and rho_scale > 0):
            raise ValueError("Invalid depth frequencies or inverse-depth scale")
        if implementation not in {"partition", "lifted160"}:
            raise ValueError(f"Unknown local geometry attention implementation: {implementation}")
        device = next(self.parameters()).device
        self.register_buffer(
            "local_depth_frequencies", torch.logspace(math.log10(depth_min), math.log10(depth_max), 8, device=device)
        )
        self.register_buffer("local_depth_normalization", torch.tensor([rho0, rho_scale], device=device))
        self.config.local_geometry_rope = dict(
            depth_min=depth_min,
            depth_max=depth_max,
            rho0=rho0,
            rho_scale=rho_scale,
            fixed_anchor=True,
            implementation=implementation,
        )
        log.info(f"Local geometry RoPE installed: implementation={implementation}")

    def install_pointflow(
        self,
        checkpoint,
        *,
        timing,
        stage=3,
        freeze_geometry=False,
        content_dim=256,
        decoder_dim=256,
        token_mode="cluster",
    ):
        """Install after base materialization/loading, before optimizer/FSDP construction.

        This explicit task-7 entry avoids losing pretrained Sonata weights to to_empty.
        Distributed initialization/resume orchestration is handled in task 9.
        """
        from cosmos_framework.model.generator.pointflow_branch import PointFlowBranch

        if self.pointflow_branch is not None:
            raise ValueError("PointFlow branch is already installed")
        parameter = next(self.parameters())
        if parameter.is_meta:
            raise ValueError("Install PointFlow on a materialized network")
        if (
            self.config.joint_attn_implementation != "two_way"
            or self.video_temporal_causal
            or not self.config.enable_fps_modulation
            or self.config.base_fps != 24
            or self.config.temporal_compression_factor_vision != 4
            or timing.steps_per_token != 4
        ):
            raise ValueError("PointFlow requires two-way, FPS modulation, base_fps=24 and temporal compression=4")
        self.pointflow_branch = PointFlowBranch(
            checkpoint,
            timing=timing,
            hidden_dim=self.hidden_size,
            stage=stage,
            freeze_geometry=freeze_geometry,
            content_dim=content_dim,
            decoder_dim=decoder_dim,
            token_mode=token_mode,
        ).to(device=parameter.device)
        self.pointflow_branch.codec.to(dtype=parameter.dtype)
        self.pointflow_branch.train(self.training)
        from cosmos_framework.utils import log

        codec = self.pointflow_branch.codec
        log.info(
            f"PointFlow branch installed: token_mode={token_mode}, stage={stage}, "
            f"freeze_geometry={freeze_geometry}, geometry_dim={codec.geometry_projection.in_features}, "
            f"geometry_motion_fusion={codec.geometry_motion_fusion}"
        )

    def install_fk(
        self, *, timing, hidden_dim=None, keypoints=21, index_dim=64, content_dim=256, decoder_dim=256, xyz_scale=1.0
    ):
        """Install after base materialization/loading, before optimizer/FSDP construction.

        The FK counterpart of ``install_pointflow``, minus the checkpoint: FK's
        encoder is learned from scratch (there is no pretrained keypoint backbone
        to lose), so this only has to happen before FSDP wraps the parameters.
        """
        from cosmos_framework.model.generator.fk_branch import FKBranch

        if self.fk_branch is not None:
            raise ValueError("FK branch is already installed")
        parameter = next(self.parameters())
        if parameter.is_meta:
            raise ValueError("Install FK on a materialized network")
        # The temporal axis is the only thing FK takes from the video timeline, so
        # these have to agree before any of it means anything.  Same preconditions
        # the PointFlow branch checks, for the same reason.
        if (
            self.config.joint_attn_implementation != "two_way"
            or self.video_temporal_causal
            or not self.config.enable_fps_modulation
            or self.config.base_fps != 24
            or self.config.temporal_compression_factor_vision != 4
            or timing.steps_per_token != 4
        ):
            raise ValueError("FK requires two-way, FPS modulation, base_fps=24 and temporal compression=4")
        self.fk_branch = FKBranch(
            timing=timing,
            hidden_dim=hidden_dim or self.hidden_size,
            keypoints=keypoints,
            index_dim=index_dim,
            content_dim=content_dim,
            decoder_dim=decoder_dim,
            xyz_scale=xyz_scale,
        ).to(device=parameter.device)
        self.fk_branch.to(dtype=parameter.dtype)
        self.fk_branch.train(self.training)
        from cosmos_framework.utils import log

        # The pointflow branch logs its install; this one did not, and the new-cluster
        # checklist asks the operator to see BOTH lines per rank.  A gate that can fail
        # without saying so is the failure mode this line removes -- and the dtype is on
        # it deliberately: ``.to(dtype=...)`` runs AFTER ``.to(device)``, and a branch
        # whose parameters missed that cast fails only at the first forward.
        log.info(
            f"FK branch installed: keypoints={keypoints}, include_anchor={self.fk_branch.include_anchor}, "
            f"index_dim={index_dim}, hidden_dim={self.fk_branch.hidden_dim}, "
            f"dtype={next(self.fk_branch.parameters()).dtype}"
        )

    def init_weights(self, buffer_device: torch.device | None):
        if self.config.vision_gen or self.config.action_gen or self.config.sound_gen:
            self.time_embedder._init_weights()

        if self.config.vision_gen:
            std = 1.0 / math.sqrt(self.patch_latent_dim)
            torch.nn.init.trunc_normal_(self.vae2llm.weight, std=std, a=-3 * std, b=3 * std)
            if self.config.enable_input_bias:
                torch.nn.init.zeros_(self.vae2llm.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2vae.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2vae.bias)

        if self.config.action_gen:
            # DomainAwareLinear uses embeddings for weights, so we initialize them differently
            # action2llm: input_size=action_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.action_dim)
            torch.nn.init.trunc_normal_(self.action2llm.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.action2llm.bias.weight)

            # llm2action: input_size=hidden_size, output_size=action_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2action.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2action.bias.weight)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.action_modality_embed, std=std, a=-3 * std, b=3 * std)

        if self.config.sound_gen:
            # sound2llm: input_size=sound_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.sound_dim)
            torch.nn.init.trunc_normal_(self.sound2llm.weight, std=std, a=-3 * std, b=3 * std)
            if self.config.enable_input_bias:
                torch.nn.init.zeros_(self.sound2llm.bias)

            # llm2sound: input_size=hidden_size, output_size=sound_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2sound.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2sound.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.sound_modality_embed, std=std, a=-3 * std, b=3 * std)

        self.language_model.init_weights(buffer_device=buffer_device)

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        return_only_new_tokens: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate text tokens using only the reasoner tower.

        Thin pass-through to ``self.language_model.generate_reasoner_text``
        (see ``unified_mot._impl_generate_reasoner_text`` for full argument
        documentation).  Handles both text-only and image-conditioned (I2V)
        prompts through this single entry point: pass
        ``pixel_values`` + ``image_grid_thw`` (and optionally
        ``attention_mask``) for image-conditioned prefill via the Qwen3-VL
        visual encoder, or omit them for text-only prefill.  Video
        conditioning is also supported via ``pixel_values_videos`` +
        ``video_grid_thw``; the image and video pairs are mutually exclusive.
        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus ``embed_tokens`` / ``norm`` / ``lm_head``; the generation pathway
        and all VFM-level multimodal embedders / heads (``vae2llm``,
        ``llm2vae``, ``sound2llm``, etc.) are bypassed.

        ``repetition_penalty`` / ``presence_penalty`` are pass-through
        sampling controls applied inside
        :func:`unified_mot._impl_generate_reasoner_text` as logit
        transformations *before* the ``do_sample`` argmax / multinomial
        branch (so they shift the greedy argmax too).  Identity defaults
        (``1.0`` / ``0.0``) keep the un-penalized fast path
        bit-identical.

        ``seed`` is a pass-through sampling-RNG knob: when provided,
        :func:`unified_mot._impl_generate_reasoner_text` allocates a
        device-local ``torch.Generator``, seeds it once with
        ``manual_seed(seed)``, and threads it into every
        ``torch.multinomial`` draw — making the decoded sequence a
        deterministic function of the seed, the prompt, and the
        penalty masks.  ``None`` (default) consumes the device's
        default RNG and is bit-identical to the pre-seed call surface.
        Has no effect under greedy decoding (the argmax branch never
        reads the generator).
        """

        return self.language_model.generate_reasoner_text(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            return_only_new_tokens=return_only_new_tokens,
        )

    def patchify_and_pack_latents(
        self, tokens_vision: torch.Tensor, token_shapes_vision: List[Tuple[int, int, int]]
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.latent_patch_size
        # Patchify and pack the latents
        packed_latent = []
        original_latent_shapes = []  # Store original shapes for unpadding later

        # C, T, H, W
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]

            # Get original latent dimensions
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))

            # Compute padded dimensions (must be divisible by p)
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p

            # Zero-pad if dimensions are not divisible by p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (self.latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )  # [C,T,H_padded,W_padded]
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded  # [C,T,H_padded,W_padded]

            # Compute number of patches after padding
            h_patches = h_padded // p
            w_patches = w_padded // p

            # Patchify
            latent = latent.reshape(
                self.latent_channel, t_actual, h_patches, p, w_patches, p
            )  # [C,T,h_patches,p,w_patches,p]
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(
                -1, p * p * self.latent_channel
            )  # [T*h_patches*w_patches,patch_latent_dim]
            packed_latent.append(latent)

        # We assumed latents we get to the network is already noised
        packed_latent = torch.cat(packed_latent, dim=0)  # [total_vision_patches,patch_latent_dim]
        return packed_latent, original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        unpatchified_latents = []

        # Split packed_mse_preds back into individual latents based on token_shapes_vision
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            # Get original shape for unpadding (if provided)
            if original_latent_shapes is not None:
                t_orig, h_orig, w_orig = original_latent_shapes[i]
                # Compute padded dimensions used during patchify
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                # Fallback: use token shapes directly (assumes no padding was needed)
                t_orig, h_orig, w_orig = t_c, h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            # noisy_frame_indexes_vision is a list of tensors, each with shape (T,),
            # where the values are the noisy frame indices.
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)

            # Initialize with the original shape (after unpadding), zeros for clean frames
            output_tensor = torch.zeros(
                (self.latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )  # [C,T,H_orig,W_orig]
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                # Extract patches for this latent
                latent_patches = packed_mse_preds[start_idx:end_idx]  # [num_patches,patch_latent_dim]
                # Reshape back to [t_n, h_patches, w_patches, p, p, channels]
                latent_patches = latent_patches.reshape(
                    t_n, h_patches, w_patches, p, p, self.latent_channel
                )  # [T_n,h_patches,w_patches,p,p,C]
                # Invert the einsum operation: "thwpqc->cthpwq"
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)  # [C,T_n,h_patches,p,w_patches,p]
                # Reshape back to [channels, t_n, h_padded, w_padded]
                latent = latent.reshape(
                    self.latent_channel, t_n, h_patches * p, w_patches * p
                )  # [C,T_n,H_padded,W_padded]

                # Crop to original dimensions (unpad the zeros)
                latent = latent[:, :, :h_orig, :w_orig]  # [C,T_n,H_orig,W_orig]

                # Fill only the noisy frame positions using the actual mask indices
                output_tensor[:, noisy_frame_indexes] = latent

                start_idx = end_idx

            unpatchified_latents.append(output_tensor.unsqueeze(0))  # [1,C,T,H,W]

        # Return list of unpatchified latents (supports variable shapes)
        return unpatchified_latents

    def pack_action(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, ...]],
        domain_id_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length action tokens into a 1D sequence for transformer input.

        Args:
            tokens_action: List of action tensors, each [T_i, action_dim] (T_i may vary).
            token_shapes_action: List of (T_i,) tuples per sample.
            domain_id_action: List of domain ID tensors, each of shape [1].

        Returns:
            Tuple of (packed_tokens, per_token_domain_id):
                packed_tokens: [total_action_tokens, action_dim]
                per_token_domain_id: [total_action_tokens]
        """
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for tokens, shape, d_id in zip(tokens_action, token_shapes_action, domain_id_action):
            T = shape[0]
            packed.append(tokens[:T])
            domain_ids.append(d_id.expand(T))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def unpack_action(
        self,
        packed_action_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, ...]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack action predictions back into per-sample action tensors.

        Args:
            packed_action_preds: Packed action predictions of shape (total_noisy_tokens, action_dim)
            token_shapes_action: Per-sample token shapes, each (T_i,) tuple.
            noisy_frame_indexes_action: List of tensors, each with shape (Tn_i,), where the values
                are the noisy frame indices for sample i.

        Returns:
            List of per-sample tensors, each of shape (T_i, action_dim), with predictions
            placed at noisy positions. Clean positions are left as zeros.
        """
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros(
                (T, self.action_dim),
                device=packed_action_preds.device,
                dtype=packed_action_preds.dtype,
            )
            t_n = len(noisy_frame_indexes)
            if t_n > 0:
                end_idx = start_idx + t_n
                output[noisy_frame_indexes] = packed_action_preds[start_idx:end_idx]
                start_idx = end_idx
            unpacked.append(output)
        return unpacked

    def pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Pack sound latents into a 1D sequence for transformer input.

        Args:
            tokens_sound: List of sound latent tensors, each [C, T]
            token_shapes_sound: List of (T, 1, 1) tuples per sample

        Returns:
            Packed tensor of shape [total_sound_tokens, C]
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            # sound: [C, T] → take first T frames → [C, T]
            # Then permute to [T, C] for packing
            sound_tokens = sound[:, :T].permute(1, 0)  # [T,C]
            packed.append(sound_tokens)
        return torch.cat(packed, dim=0)  # [total_sound_tokens,C]

    def unpack_sound_latents(
        self,
        packed_sound_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack sound predictions back into per-sample sound latents.

        Args:
            packed_sound_preds: Packed sound predictions of shape (total_noisy_tokens, sound_dim)
            token_shapes_sound: List of (T, 1, 1) tuples per sample
            noisy_frame_indexes_action: List of tensors, each with shape (T_i,), where the values
                are the noisy frame indices. T_i <= max_T.

        Returns:
            List of per-sample tensors, each [C, T], with predictions placed at noisy positions.
            Clean positions are left as zeros.
        """
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            # Initialize output with zeros for clean positions
            output = torch.zeros(
                (self.sound_dim, T),
                device=packed_sound_preds.device,
                dtype=packed_sound_preds.dtype,
            )

            t_n = len(noisy_frame_indexes)

            if t_n > 0:
                end_idx = start_idx + t_n
                # packed_sound_preds: [total_noisy_tokens, C] → transpose and fill at noisy positions
                output[:, noisy_frame_indexes] = packed_sound_preds[
                    start_idx:end_idx
                ].T  # packed_sound_preds[...]: [T_n,C] → .T: [C,T_n]
                start_idx = end_idx

            unpacked.append(output)
        return unpacked

    def _encode_text(
        self,
        packed_seq: PackedSequence,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens and initialize packed_sequence.

        Args:
            packed_seq: PackedSequence containing text_ids and text_indexes.

        Returns:
            tuple of (packed_sequence, target_dtype) where packed_sequence has text embeddings filled in.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)  # [N_text,hidden_size]
        packed_sequence = packed_text_embedding.new_zeros(
            size=(packed_seq.sequence_length, self.hidden_size)
        )  # [N_total,hidden_size]
        packed_sequence[packed_seq.text_indexes] = (
            packed_text_embedding  # [N_text,hidden_size] scattered into [N_total,hidden_size]
        )
        return packed_sequence, packed_text_embedding.dtype

    def _embed_packed_timesteps(self, timesteps: torch.Tensor, packed_seq: PackedSequence) -> torch.Tensor:
        """Embed noised-token timesteps, reusing work when packing proves they share one scalar."""
        if packed_seq.uses_single_timestep and timesteps.numel() > 1:
            timestep = timesteps[:1]  # [1]
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                timestep_embed = self.time_embedder(timestep)  # [1,hidden_size]
            # Materialize: expand() aliases storage; in-place ops on a float32 no-op .to() would corrupt all rows.
            return timestep_embed.expand(timesteps.shape[0], -1).contiguous()  # [N_noisy_frames,hidden_size]

        # Timesteps are computed in FP32 for numerical stability.
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            return self.time_embedder(timesteps)  # [N_noisy_frames,hidden_size]

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing vision tokens and metadata.
            packed_sequence: The packed sequence tensor to fill vision embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).

        Returns:
            Original latent shapes before padding (for unpadding during decode), or None if no vision tokens.
        """
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            # No vision tokens in this batch
            return None

        vision = packed_seq.vision
        assert vision.tokens is not None  # Type narrowing (checked above but reassignment loses it)
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        torch._assert(
            vision.timesteps.dtype in (torch.long, torch.float32),
            f"Timestep must be long/float32, got {vision.timesteps.dtype}",
        )

        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            vision.tokens, vision.token_shapes
        )  # packed_tokens_vision: [total_vision_patches,patch_latent_dim]
        packed_tokens_vision = self.vae2llm(packed_tokens_vision.to(target_dtype))  # [total_vision_patches,hidden_size]

        has_noisy_vision = vision.mse_loss_indexes.numel() > 0

        if has_noisy_vision:
            timesteps_vision = vision.timesteps.to(dtype=torch.float32) * self.timestep_scale  # [N_noisy_frames_vision]

            packed_timestep_embeds_vision = self._embed_packed_timesteps(
                timesteps_vision, packed_seq
            )  # [N_noisy_frames_vision,hidden_size]
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(
                target_dtype
            )  # [N_noisy_frames_vision,hidden_size]

            packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )  # [total_vision_patches,hidden_size]

        packed_sequence[vision.sequence_indexes] = (
            packed_tokens_vision  # [total_vision_patches,hidden_size] scattered into [N_total,hidden_size]
        )
        return original_latent_shapes

    def _decode_vision(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Decode vision tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing mse_loss_indexes_vision and token_shapes_vision.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with mse_preds (modified in-place).
            original_latent_shapes: Original latent shapes before padding (for unpadding).
        """
        vision = packed_seq.vision
        # Check if no vision or no noisy vision tokens
        has_noisy_vision = (
            vision is not None
            and vision.tokens is not None
            and isinstance(vision.mse_loss_indexes, torch.Tensor)
            and vision.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_vision:
            # No noisy vision tokens present. The model is predicting actions
            # given clean vision tokens. We need to execute a dummy forward to maintain
            # computation graph consistency across ranks (FSDP should torch all weights).
            preds_vision = torch.zeros(
                [1, self.patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,patch_latent_dim]
            preds_vision = self.vae2llm(preds_vision)  # [1,hidden_size]
            preds_vision = self.llm2vae(preds_vision)  # [1,patch_latent_dim]
            # Return a list of per-sample zero tensors with correct shapes (e.g. (C, T, H, W)),
            # so downstream code (_get_velocity, _compute_flow_matching_loss) that iterates over preds_vision
            # gets properly-shaped tensors. Without this, the dummy tensor (1, patch_latent_dim)
            # would cause a size mismatch when concatenating vision+action velocities.
            # When vision is None (no vision in batch), fall back to [preds_vision] purely for
            # gradient graph consistency — it won't be iterated over.
            if vision is not None and vision.tokens is not None:
                preds_vision_list = [torch.zeros_like(tok) for tok in vision.tokens]
                # Inject dummy forward's computation graph so vae2llm/llm2vae params
                # stay in the autograd graph (zeros_like creates detached tensors).
                preds_vision_list[0] = preds_vision_list[0] + 0.0 * preds_vision.sum()
            else:
                preds_vision_list = [preds_vision]
            output_dict.update(preds_vision=preds_vision_list)
        else:
            assert vision is not None  # Type narrowing
            assert isinstance(vision.mse_loss_indexes, torch.Tensor)
            assert vision.noisy_frame_indexes is not None
            preds_vision = self.llm2vae(
                last_hidden_state[vision.mse_loss_indexes]
            )  # [total_noisy_vision_patches,patch_latent_dim]
            preds_vision = self.unpatchify_and_unpack_latents(
                preds_vision,
                token_shapes_vision=vision.token_shapes,
                noisy_frame_indexes_vision=vision.noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
            output_dict.update(preds_vision=preds_vision)

    def _encode_action(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Encode action tokens and fill into packed_sequence."""
        if packed_seq.action is None or packed_seq.action.tokens is None:
            # No action tokens in this batch
            return

        action: ModalityData = packed_seq.action
        assert action.token_shapes is not None
        assert isinstance(action.sequence_indexes, torch.Tensor)
        assert isinstance(action.timesteps, torch.Tensor)
        assert isinstance(action.mse_loss_indexes, torch.Tensor)

        # Pack variable-length action tokens into a 1D sequence (same pattern as pack_sound_latents)
        packed_tokens_action, per_token_domain_id = self.pack_action(
            action.tokens, action.token_shapes, action.domain_id
        )
        packed_tokens_action = self.action2llm(packed_tokens_action, per_token_domain_id)

        packed_tokens_action = packed_tokens_action + self.action_modality_embed.view(
            1, -1
        )  # [B_action*T_action,hidden_size]

        has_noisy_actions = has_noisy_tokens(action)
        if has_noisy_actions:
            timesteps_action = action.timesteps * self.timestep_scale  # [N_noisy_frames_action]
            packed_timestep_embeds_action = self._embed_packed_timesteps(
                timesteps_action, packed_seq
            )  # [N_noisy_frames_action,hidden_size]
            packed_timestep_embeds_action = packed_timestep_embeds_action.to(
                target_dtype
            )  # [N_noisy_frames_action,hidden_size]

            packed_tokens_action = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_action,
                packed_timestep_embeds=packed_timestep_embeds_action,
                noisy_frame_indexes=action.noisy_frame_indexes,
                token_shapes=action.token_shapes,
            )  # [B_action*T_action,hidden_size]

        packed_sequence[action.sequence_indexes] = (
            packed_tokens_action  # [B_action*T_action,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_action(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode action tokens from hidden states and update output_dict."""
        action = packed_seq.action
        # Shared predicate with OmniMoTModel's has_noisy_actions gating: actions
        # are decodable targets only when the packer marked action tokens noisy.
        has_noisy_action = has_noisy_tokens(action)
        if not has_noisy_action:
            # dummy forward to maintain computation graph consistency across ranks
            preds_action = torch.zeros(
                [1, self.action_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,action_dim]
            dummy_domain_id = torch.zeros([1], device=last_hidden_state.device, dtype=torch.long)  # [1]
            preds_action = self.action2llm(preds_action, dummy_domain_id) + self.action_modality_embed.view(
                1, -1
            )  # [1,hidden_size]
            preds_action = self.llm2action(preds_action, dummy_domain_id)  # [1,action_dim]
            # Return a list of per-sample zero tensors with correct shapes (e.g. (T, action_dim)),
            # so downstream code (_get_velocity, _compute_flow_matching_loss) that iterates over preds_action
            # gets properly-shaped tensors. Without this, the dummy tensor (1, action_dim)
            # would cause a size mismatch when concatenating vision+action velocities.
            if action is not None and action.tokens is not None:
                preds_action_list = [torch.zeros_like(tok) for tok in action.tokens]
                # Inject dummy forward's computation graph so DomainAwareLinear params
                # stay in the autograd graph (zeros_like creates detached tensors).
                preds_action_list[0] = preds_action_list[0] + 0.0 * preds_action.sum()
            # When action is None (no action in batch), fall back to [preds_action] purely for
            # gradient graph consistency — it won't be iterated over.
            else:
                preds_action_list = [preds_action]
            output_dict.update(preds_action=preds_action_list)
        else:
            assert action is not None  # Type narrowing
            assert isinstance(action.mse_loss_indexes, torch.Tensor)
            assert action.condition_mask is not None
            assert len(action.domain_id) > 0

            action_hidden_states = last_hidden_state[action.mse_loss_indexes]  # [total_noisy_action_tokens,hidden_size]

            # Build per-token domain IDs for the noisy tokens (same expansion logic as pack_action)
            domain_ids: list[torch.Tensor] = []
            for nfi, d_id in zip(action.noisy_frame_indexes, action.domain_id):
                domain_ids.append(d_id.expand(len(nfi)))
            per_token_domain_id = torch.cat(domain_ids, dim=0)

            preds_action = self.llm2action(
                action_hidden_states, per_token_domain_id
            )  # [total_noisy_action_tokens,action_dim]
            preds_action = self.unpack_action(preds_action, action.token_shapes, action.noisy_frame_indexes)
            output_dict.update(preds_action=preds_action)

    def _encode_sound(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Encode sound tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing sound tokens and metadata.
            packed_sequence: The packed sequence tensor to fill sound embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            # No sound tokens in this batch
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        # Pack sound latents: list of [C, T] tensors → [total_tokens, C]
        packed_tokens_sound = self.pack_sound_latents(
            sound.tokens, sound.token_shapes
        )  # [total_sound_tokens,sound_dim]
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)  # [total_sound_tokens,sound_dim]

        # Project sound tokens + modality embedding. Position info comes from
        # mRoPE position IDs in the attention layers.
        packed_tokens_sound = (
            self.sound2llm(packed_tokens_sound) + self.sound_modality_embed
        )  # [total_sound_tokens,hidden_size]

        has_noisy_sound = sound.mse_loss_indexes.numel() > 0
        if has_noisy_sound:
            timesteps_sound = sound.timesteps * self.timestep_scale  # [N_noisy_frames_sound]
            packed_timestep_embeds_sound = self._embed_packed_timesteps(
                timesteps_sound, packed_seq
            )  # [N_noisy_frames_sound,hidden_size]
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(
                target_dtype
            )  # [N_noisy_frames_sound,hidden_size]

            packed_tokens_sound = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )  # [total_sound_tokens,hidden_size]

        packed_sequence[sound.sequence_indexes] = (
            packed_tokens_sound  # [total_sound_tokens,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_sound(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode sound tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing sound modality data.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with preds_sound (modified in-place).
        """
        sound = packed_seq.sound
        # Check if no sound or no noisy sound tokens
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_sound:
            # dummy forward to maintain computation graph consistency across ranks
            preds_sound = torch.zeros(
                [1, self.sound_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,sound_dim]
            preds_sound = self.sound2llm(preds_sound) + self.sound_modality_embed  # [1,hidden_size]
            preds_sound = self.llm2sound(preds_sound)  # [1,sound_dim]
            if sound is not None and sound.tokens is not None:
                preds_sound_list = [torch.zeros_like(tok) for tok in sound.tokens]
                preds_sound_list[0] = preds_sound_list[0] + 0.0 * preds_sound.sum()
            else:
                preds_sound_list = [preds_sound]
            output_dict.update(preds_sound=preds_sound_list)
        else:
            assert sound is not None  # Type narrowing
            assert isinstance(sound.mse_loss_indexes, torch.Tensor)
            assert sound.condition_mask is not None
            preds_sound = self.llm2sound(
                last_hidden_state[sound.mse_loss_indexes]
            )  # [total_noisy_sound_tokens,sound_dim]
            preds_sound = self.unpack_sound_latents(
                preds_sound, sound.token_shapes, sound.noisy_frame_indexes
            )  # list of [C,T] per sample
            output_dict.update(preds_sound=preds_sound)

    def forward(
        self,
        packed_seq: PackedSequence,
        memory: MemoryState | None = None,
        video_temporal_causal: bool | None = None,
        pointflow_displacement: torch.Tensor | None = None,
        pointflow_sigma: torch.Tensor | None = None,
        fk_displacement: torch.Tensor | None = None,
        fk_sigma: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass for Cosmos3VFMNetwork.

        Args:
            packed_seq: PackedSequence containing all packed tensors and metadata.
                See PackedSequence dataclass for field details.
            memory: Optional MemoryState for persistent KV-cache memory
                (AR inference or rolling-KV-cache training).  Built by
                ``OmniMoTModel.build_memory_state()``.
            pointflow_displacement: Explicit noisy point displacement [H,sum(N),3].
            pointflow_sigma: Diffusion time [B], separate from physical frame time.
            video_temporal_causal: Per-call attention-mode override; ``None``
                (default) uses the config-selected ``self.video_temporal_causal``.

        Returns:
            dict with keys:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity predictions for action tokens (if action_gen).
                - "preds_sound": Velocity predictions for sound tokens (if sound_gen).
                - "preds_pointflow": Original-point velocity [H,sum(N),3], when PointFlow is supplied.
                - "last_hidden_state": Last hidden state from the transformer.
                - "lbl_metadata_*": Load balancing metadata.
                - "ce_preds": Cross-entropy predictions (if predict_text_tokens is True).
        """
        # Note: During inference with @torch.no_grad(), model may be in training mode
        # This is intentional for proper batch norm / dropout behavior
        # assert self.training, "Cosmos3VFMNetwork only supports training mode"

        fk_seen = packed_seq.fk is not None or packed_seq.fk_data is not None
        if fk_seen and (
            self.config.joint_attn_implementation != "two_way"
            or (self.video_temporal_causal if video_temporal_causal is None else video_temporal_causal)
            or memory is not None
            or packed_seq.control_weights is not None
            or (self.parallel_dims is not None and self.parallel_dims.cp_enabled)
            or self.pad_for_cuda_graphs
        ):
            raise ValueError("FK tokens require CP=1 two-way, no temporal causal/memory/controls/CUDA graphs")

        if packed_seq.fk_data is not None:
            if self.fk_branch is None:
                raise ValueError("FK data supplied but no FK branch is installed")
            if fk_displacement is None or fk_sigma is None:
                raise ValueError("Supply explicit noisy FK displacement and sigma; never use GT by default")
            encoded = self.fk_branch.encode(packed_seq.fk_data, fk_displacement, fk_sigma)
            if "anchor_uv" in packed_seq.fk_data.inputs:
                from cosmos_framework.model.generator.pointflow_branch import video_aligned_point_positions

                if packed_seq.pointflow_data is None:
                    raise ValueError("Projected FK requires PointFlow canvas metadata")
                fk_geometry = {
                    "cluster_uv": packed_seq.fk_data.inputs["anchor_uv"],
                    "cluster_batch": packed_seq.fk_data.inputs["point_batch"],
                }
                positions = video_aligned_point_positions(
                    packed_seq,
                    packed_seq.pointflow_data.inputs,
                    fk_geometry,
                    self.fk_branch.timing,
                    self.latent_downsample,
                    check_grid=False,
                )
            else:
                positions = fk_positions(packed_seq, packed_seq.fk_data.inputs["point_batch"], self.fk_branch.timing)
            packed_seq = attach_fk_tokens(packed_seq, encoded, positions)
        elif fk_displacement is not None or fk_sigma is not None:
            raise ValueError("FK state supplied without FK inputs")

        pointflow_geometry = None
        if packed_seq.point is not None or packed_seq.pointflow_data is not None:
            if (
                self.config.joint_attn_implementation != "two_way"
                or (self.video_temporal_causal if video_temporal_causal is None else video_temporal_causal)
                or memory is not None
                or packed_seq.control_weights is not None
                or (self.parallel_dims is not None and self.parallel_dims.cp_enabled)
                or self.pad_for_cuda_graphs
            ):
                raise ValueError("Point tokens require CP=1 two-way, no temporal causal/memory/controls/CUDA graphs")

        if packed_seq.pointflow_data is not None:
            if self.pointflow_branch is None:
                raise ValueError("PointFlow data supplied but no PointFlow branch is installed")
            if pointflow_displacement is None or pointflow_sigma is None:
                raise ValueError("Supply explicit noisy PointFlow displacement and sigma; never use GT by default")
            packed_seq, pointflow_geometry, pointflow_displacement = self.pointflow_branch.encode(
                packed_seq,
                pointflow_displacement,
                pointflow_sigma,
                pixel_stride=self.latent_downsample,
            )
        elif pointflow_displacement is not None or pointflow_sigma is not None:
            raise ValueError("PointFlow state supplied without PointFlow inputs")

        packed_sequence, target_dtype = self._encode_text(packed_seq)  # packed_sequence: [N_total,hidden_size]

        if packed_seq.fk is not None:
            packed_sequence[packed_seq.fk.sequence_indexes] = packed_seq.fk.tokens.to(
                device=packed_sequence.device, dtype=target_dtype
            )

        if packed_seq.point is not None:
            packed_sequence[packed_seq.point.sequence_indexes] = packed_seq.point.tokens.to(
                device=packed_sequence.device, dtype=target_dtype
            )

        # encode vision tokens
        original_latent_shapes: List[Tuple[int, int, int]] | None = None
        if self.config.vision_gen:
            original_latent_shapes = self._encode_vision(packed_seq, packed_sequence, target_dtype)

        # encode action tokens
        if self.config.action_gen:
            self._encode_action(packed_seq, packed_sequence, target_dtype)

        # encode sound tokens
        if self.config.sound_gen:
            self._encode_sound(packed_seq, packed_sequence, target_dtype)

        assert packed_seq.attn_modes is not None
        assert packed_seq.split_lens is not None
        use_video_temporal_causal = (
            self.video_temporal_causal if video_temporal_causal is None else video_temporal_causal
        )

        # Get all generation sequence indexes for MoE routing
        # IMPORTANT: Include ALL latent tokens (video + action + sound), not just generation targets.
        # Condition tokens still need to be routed to diffusion experts; they are excluded from
        # LOSS computation, not from routing.
        all_gen_indexes = []
        if packed_seq.vision is not None:
            assert packed_seq.vision.token_shapes is not None
            assert isinstance(packed_seq.vision.sequence_indexes, torch.Tensor)
            all_gen_indexes.append(packed_seq.vision.sequence_indexes)
        if packed_seq.action is not None and isinstance(packed_seq.action.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.action.sequence_indexes)
        if packed_seq.sound is not None and isinstance(packed_seq.sound.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.sound.sequence_indexes)
        if packed_seq.fk is not None:
            all_gen_indexes.append(packed_seq.fk.sequence_indexes)
        if packed_seq.point is not None:
            all_gen_indexes.append(packed_seq.point.sequence_indexes)
        vision_sequence_indexes = torch.cat(all_gen_indexes, dim=0) if all_gen_indexes else None  # [N_gen_tokens]

        # When temporal causal is enabled the buffer is [action_t0, vision_t0, action_t1, vision_t1, ...].
        # After torch.cat([vision_indexes, action_indexes]) the interleaved order is lost; sorting restores it.
        if self.video_temporal_causal:
            assert packed_seq.sound is None, "Sound generation is not supported with video_temporal_causal=True."
            if vision_sequence_indexes is not None:
                vision_sequence_indexes = vision_sequence_indexes.sort().values  # [N_gen_tokens]

        vision_token_shapes = packed_seq.vision.token_shapes if packed_seq.vision else None

        # The packer is the single source of truth for the supertoken layout.
        # ``num_action_tokens_per_supertoken`` is stamped onto ``packed_seq`` by
        # ``pack_supertokens_temporal_causal`` (= tcf when actions are packed
        # inline, 0 otherwise) and read unchanged by the attention builder, the
        # NATTEN metadata generator, and the rolling KV-cache state — keeping
        # all downstream supertoken geometry automatically in sync with the pack.
        num_action_tokens_per_supertoken = packed_seq.num_action_tokens_per_supertoken

        replicated_attention_io_cp = (
            self.attention_io_layout == "replicated"
            and self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
        )
        # ``sequence_sharded`` attention I/O shards the token sequence, so
        # packing must pad sequence lengths to the CP size and the input/output
        # sequence helpers need the CP mesh.  ``replicated`` attention I/O keeps
        # current-frame sequences replicated and uses the CP mesh later inside
        # attention to slice local heads, so the effective sequence-sharding
        # world size is 1 here.
        sequence_shard_parallel_dims = None if replicated_attention_io_cp else self.parallel_dims
        sequence_shard_world_size = (
            1 if replicated_attention_io_cp else (self.parallel_dims.cp_size if self.parallel_dims else 1)
        )

        input_pack, attention_meta, natten_metadata_list = build_packed_sequence(
            self.config.joint_attn_implementation,
            packed_sequence=packed_sequence,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
            packed_und_token_indexes=packed_seq.text_indexes,
            packed_gen_token_indexes=vision_sequence_indexes,
            num_heads=self.num_heads,
            is_image_batch=packed_seq.is_image_batch,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            token_shapes=packed_seq.vision.token_shapes,
            natten_parameter_list=self.natten_parameter_list,
            cp_world_size=sequence_shard_world_size,
            video_temporal_causal=use_video_temporal_causal,
            skip_natten_metadata=memory is not None and not memory.requires_natten_metadata(),
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=packed_seq.action.token_shapes if packed_seq.action else None,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            pad_for_cuda_graphs=self.pad_for_cuda_graphs,
        )

        # Keyed off the payload rather than the environment, so the modality map
        # is built exactly when the pairwise reference attention will consume it.
        # Under the default legacy_mrope this stays None and the generator uses
        # dispatch_attention_fn with the plain full mRoPE.
        if packed_seq.point is not None and packed_seq.point.attention_mode == "pairwise_point_mrope":
            modalities = torch.zeros(packed_seq.sequence_length, dtype=torch.long, device=packed_sequence.device)
            if packed_seq.action is not None:
                modalities[packed_seq.action.sequence_indexes] = 1
            modalities[packed_seq.point.sequence_indexes] = 2
            attention_meta.pointflow_modalities = modalities

        if hasattr(self, "local_depth_frequencies"):
            from cosmos_framework.model.generator.pointflow_fk_attention import make_partition

            if (
                sequence_shard_world_size != 1
                or memory is not None
                or self.pad_for_cuda_graphs
                or packed_seq.control_weights is not None
                or packed_seq.null_action_supertokens
                or packed_seq.attn_modes != ["causal", "full"] * len(packed_seq.sample_lens)
            ):
                raise ValueError("Local geometry RoPE requires CP=1, two-way, no memory/control/null-action/cudagraph")
            if packed_seq.point is None or packed_seq.fk is None or "anchor_uv" not in packed_seq.fk_data.inputs:
                raise ValueError("Local geometry RoPE requires projected FK and PointFlow payloads")
            if not self.pointflow_branch.codec.include_anchor or not self.fk_branch.include_anchor:
                raise ValueError("Local geometry RoPE currently requires anchor tokens")
            indexes, depths = [], []
            for payload, xyz in (
                (packed_seq.point, pointflow_geometry["cluster_xyz"]),
                (packed_seq.fk, packed_seq.fk_data.inputs["anchor_xyz"]),
            ):
                z = xyz[:, 2].float()
                if not torch.isfinite(z).all() or (z <= 0).any():
                    raise ValueError("Local geometry RoPE requires finite positive anchor camera Z")
                # noisy_indexes retains [block, original point/cluster] order even
                # after the other modality remaps the packed sequence.
                ids = payload.noisy_indexes
                anchor_ids = ids[0] - xyz.shape[0]  # overwritten sample-wise below
                # Anchor tokens immediately precede block 0 within each sample.
                counts = (
                    pointflow_geometry["cluster_offsets"]
                    if payload is packed_seq.point
                    else packed_seq.fk_data.inputs["point_offsets"]
                )
                start = 0
                for end in counts.tolist():
                    anchor_ids[start:end] = ids[0, start:end] - (end - start)
                    start = end
                indexes.append(torch.cat((anchor_ids[None], ids)).flatten())
                rho = (z.reciprocal() - self.local_depth_normalization[0]) / self.local_depth_normalization[1]
                depths.append(rho[None].expand(ids.shape[0] + 1, -1).flatten())
            attention_meta.local_geometry_partition = make_partition(
                packed_seq.sample_lens, packed_seq.split_lens, torch.cat(indexes), torch.cat(depths)
            )
            attention_meta.local_depth_frequencies = self.local_depth_frequencies
            attention_meta.local_geometry_implementation = self.config.local_geometry_rope.get(
                "implementation", "partition"
            )
            packed_seq.point.attention_mode = "local_pointflow_fk_mrope"

        # ── Multi-control transfer: annotate SplitInfo with per-item ranges ──────
        # Activated only when packed_seq carries control_weights, i.e. the caller
        # has set up a multi-control batch via build_transfer_batch.
        #
        # multi_control_two_way_attention runs N independent maskless SDPA passes,
        # one per control.  For each pass i, KV = [text | ctrl_i | noisy].
        # The final noisy output is the weighted sum of the N pass outputs:
        #   noisy_out = w_1 * noisy_out_1 + ... + w_N * noisy_out_N
        # All SDPA calls are maskless → Flash Attention always active.
        # N=1, w=1.0 → identical to two_way_attention.
        #
        # CP compatibility: control_stream_token_ranges are gen-relative global
        # offsets computed here, before CP sharding.  Ulysses CP restores the full
        # sequence on every rank (via all-to-all) before calling dispatch_attention,
        # so the global ranges are valid indices inside multi_control_two_way_attention.
        if (
            isinstance(attention_meta, SplitInfo)
            and packed_seq.control_weights is not None
            and packed_seq.vision_item_split_lens
        ):
            # For multi-control, each sample must have N controls + 1 noisy item
            # (items 0..N-2 are controls, item N-1 is the noisy target).
            # Only batch_size=1 is supported; assert to catch misuse early.
            assert len(packed_seq.vision_item_split_lens) == 1, (
                f"Multi-control transfer requires batch_size=1, got {len(packed_seq.vision_item_split_lens)} samples."
            )
            item_lens = packed_seq.vision_item_split_lens[0]  # [L_ctrl0, L_ctrl1, ..., L_noisy]
            weights = packed_seq.control_weights[0]  # [w_ctrl0, w_ctrl1, ...]
            assert len(item_lens) > 1, (
                f"Multi-control requires at least 1 control + 1 noisy item; got vision_item_split_lens={item_lens}."
            )
            assert len(weights) == len(item_lens) - 1, (
                f"control_weights length ({len(weights)}) must equal number of control items ({len(item_lens) - 1})."
            )
            ctrl_ranges: list[tuple[int, int]] = []
            cursor = 0
            for lens in item_lens[:-1]:  # all but last = control streams
                ctrl_ranges.append((cursor, cursor + lens))
                cursor += lens
            noisy_range = (cursor, cursor + item_lens[-1])
            n_gen = int(vision_sequence_indexes.shape[0]) if vision_sequence_indexes is not None else 0
            assert noisy_range[1] == n_gen, (
                f"vision_item_split_lens sums to {noisy_range[1]} gen tokens but packed tensor has "
                f"{n_gen}; packing inconsistency detected."
            )
            attention_meta.control_stream_token_ranges = ctrl_ranges
            attention_meta.noisy_token_range = noisy_range
            attention_meta.control_weights = weights

        input_pack, packed_position_ids = get_context_parallel_sharded_sequence(
            attn_implementation=self.config.joint_attn_implementation,
            input_pack=input_pack,
            position_ids=packed_seq.position_ids,
            parallel_dims=sequence_shard_parallel_dims,
        )

        packed_outputs, lbl_metadata = self.language_model(
            input_pack,
            attention_mask=attention_meta,
            position_ids=packed_position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        last_hidden_state = get_context_parallel_last_hidden_state(
            packed_outputs=packed_outputs,
            parallel_dims=sequence_shard_parallel_dims,
        )  # [N_total,hidden_size]
        output_dict = dict()
        if packed_seq.point is not None:
            output_dict["point_hidden"] = packed_seq.point.hidden(last_hidden_state)

        if pointflow_geometry is not None:
            output_dict["preds_pointflow"] = self.pointflow_branch.decode(
                pointflow_geometry,
                packed_seq.pointflow_data.inputs,
                output_dict["point_hidden"],
                pointflow_displacement,
                pointflow_sigma,
            )
            output_dict["pointflow_cluster_offsets"] = pointflow_geometry["cluster_offsets"]
            output_dict["pointflow_point_offsets"] = pointflow_geometry["point_offsets"]
        if packed_seq.fk is not None:
            output_dict["fk_hidden"] = packed_seq.fk.hidden(last_hidden_state)

        if packed_seq.fk_data is not None:
            output_dict["preds_fk"] = self.fk_branch.decode(
                output_dict["fk_hidden"],
                fk_displacement,
                fk_sigma,
                packed_seq.fk_data.inputs["point_batch"],
                len(packed_seq.fk_data.labeled),
            )
            output_dict["fk_point_offsets"] = packed_seq.fk_data.inputs["point_offsets"]

        # decode vision tokens
        if self.config.vision_gen:
            self._decode_vision(packed_seq, last_hidden_state, output_dict, original_latent_shapes)

        # decode action tokens
        if self.config.action_gen:
            self._decode_action(packed_seq, last_hidden_state, output_dict)

        # decode sound tokens
        if self.config.sound_gen:
            self._decode_sound(packed_seq, last_hidden_state, output_dict)

        output_dict.update(last_hidden_state=last_hidden_state)
        for lbl_metadata_key, lbl_metadata_value in lbl_metadata.items():
            output_dict.update({f"lbl_metadata_{lbl_metadata_key}": lbl_metadata_value})
        if self.predict_text_tokens:
            packed_ce_preds = self.language_model.lm_head(
                last_hidden_state[packed_seq.ce_loss_indexes]
            )  # [N_ce_tokens,vocab_size]
            output_dict["ce_preds"] = packed_ce_preds

        return output_dict


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Apply timestep embeddings to noisy tokens.
    Tn is the number of noisy frames for a given sample.
    Tc is the number of clean frames for a given sample.
    T is the total number of frames for a given sample.
    T = Tn + Tc

    Args:
        packed_tokens: The packed tokens to apply timestep embeddings to.
        packed_timestep_embeds: The packed timestep embeddings to apply.
        noisy_frame_indexes: The frame indices to apply timestep embeddings to
            (list of tensors, each with shape (Tn,)).
        token_shapes: The token shapes for each sample. Each entry is a tuple
            shaped like ``(T, ...)`` where trailing dimensions represent the spatial grid.

    Returns:
        The packed tokens with timestep embeddings applied to the noisy tokens.
    """

    # Handle variable token shapes by processing each sample's noisy_frame_indexes individually.
    # The noisy indices are first expanded to cover the entire spatial grid of each frame.
    #
    # For video frames, the spatial grid is (H, W).
    # For action frames, the spatial grid is ().
    # For sound frames, the spatial grid is (1, 1).
    #
    # The noisy indices are then flattened into a single tensor overall. When flattening,
    # we must ensure that the noisy indices from each sample are unique by adding the
    # cumulative sum of the token shapes of previous samples to the noisy indices for
    # a given sample.
    start_noisy_index = 0
    flattened_noisy_frame_indexes = []

    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
        assert noisy_indexes_i.numel() <= token_shape_i[0]
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)  # [spatial_numel_i]
        noisy_indexes_i = (
            (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
        )  # [Tn_i,spatial_numel_i]
        noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index  # [Tn_i,spatial_numel_i]
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())  # [Tn_i*spatial_numel_i]
        start_noisy_index += math.prod(token_shape_i)

    flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)  # [total_noisy_patches]

    assert packed_tokens.dim() == 2
    assert packed_timestep_embeds.dim() == 2
    assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
    assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
    assert flattened_noisy_frame_indexes.dim() == 1
    assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
        -1,
        packed_tokens.shape[1],
    )  # [total_noisy_patches,hidden_size]

    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )  # [total_tokens,hidden_size]
