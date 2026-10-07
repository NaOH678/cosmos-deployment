---
license: other
license_name: openmdw1.1-license
license_link: >-
  https://openmdw.ai/license/1-1/
library_name: cosmos
tags:
  - nvidia
  - cosmos
  - cosmos3
  - world action model
  - policy model
countDownloads:
  - checkpoint.json
  - config.json
  - generation_config.json
  - model.safetensors.index.json
  - model_index.json
  - tokenizer.json
  - tokenizer_config.json
  - sound_tokenizer/config.json
  - sound_tokenizer/diffusion_pytorch_model.safetensors
  - text_tokenizer/tokenizer.json
  - text_tokenizer/tokenizer_config.json
  - transformer/config.json
  - transformer/diffusion_pytorch_model-00001-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00002-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00003-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00004-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00005-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00006-of-00007.safetensors
  - transformer/diffusion_pytorch_model-00007-of-00007.safetensors
  - transformer/diffusion_pytorch_model.safetensors.index.json
  - vae/config.json
  - vae/diffusion_pytorch_model.safetensors
  - vision_encoder/config.json
  - vision_encoder/model.safetensors  
---

# **Cosmos 3: Omnimodal World Models for Physical AI**
**[Model Collection](https://huggingface.co/collections/nvidia/cosmos3)** | **[Code](https://github.com/nvidia/cosmos)** | **[White Paper](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)** | **[Website](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)**

[NVIDIA Cosmos™](https://github.com/nvidia/cosmos) is a world foundation model platform designed to accelerate the development of Physical AI by enabling machines to understand, simulate, and interact with the physical world across robotics, autonomous driving, and smart space environments, including industrial and factory-scale applications.

# Model Overview: Cosmos3-Edge-Policy-DROID

## Description

Cosmos3 is a collection of Omnimodal world models capable of generating dynamic, high-quality video, image, audio, and action commands from combinations of text, image, video, and action trajectory inputs. It serves as a foundational building block for a broad range of Physical AI applications and research spanning world understanding, world generation, simulation, and embodied policy learning.

This model is ready for commercial and non-commercial use.

**Model Developer:** NVIDIA

### Model Versions

**Released on: 07/20/2026**
- Cosmos3-Edge:
  - Given multimodal inputs including text, images, video, and action trajectories, generate coherent text, images, video, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Edge-Policy-DROID:
  - Given language instructions and visual observations from the DROID robot platform, generate robot action trajectories for manipulation and control tasks.

- Cosmos3-Super-Image2Video-4Step:
  - Given one or more input images and optional text instructions, generate temporally coherent video sequences that are consistent with the provided visual content. Distilled from Cosmos3-Super-Image2Video using [Improved Distribution Matching Distillation (DMD2)](https://arxiv.org/abs/2405.14867), enabling high-quality generation in 4 steps.

- Cosmos3-Super-Text2Image-4Step:
  - Given text input, generate high-fidelity images that are consistent with the provided description. Distilled from Cosmos3-Super-Text2Image using [Improved Distribution Matching Distillation (DMD2)](https://arxiv.org/abs/2405.14867), enabling high-quality generation in 4 steps.

**Released on: 05/31/2026**
- Cosmos3-Nano:
  - Given multimodal inputs including text, images, video, audio, and action trajectories, generate coherent text, images, video, audio, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Super:
  - Given multimodal inputs including text, images, video, audio, and action trajectories, generate coherent text, images, video, audio, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Nano-Policy-DROID:
  - Given language instructions and visual observations from the DROID robot platform, generate robot action trajectories for manipulation and control tasks.

- Cosmos3-Super-Image2Video:
  - Given one or more input images and optional text instructions, generate temporally coherent video sequences that are consistent with the provided visual content.

- Cosmos3-Super-Text2Image:
  - Given text input, generate high-fidelity images that are consistent with the provided description.

### License

This model is released under the [OpenMDW1.1](https://openmdw.ai/license/1-1/)

### Deployment Geography

Global

### Use Case

Physical AI: Encompassing robotics, autonomous vehicles (AV), and smart space environments, including industrial and factory-scale applications.

### Release Date

Hugging Face 07/20/2026 via [https://huggingface.co/collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
GitHub 07/20/2026 via [https://github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)

## Model Architecture

**Architecture Type:** Transformer

**Network Architecture:** Mixture-of-Transformers (MoT)

Cosmos3 is an Omni-modal foundation model built on a Mixture-of-Transformers (MoT) architecture consisting of two complementary transformer towers: an autoregressive transformer for discrete token generation and a diffusion transformer for continuous multimodal generation. During inference, text is generated through standard next-token autoregressive decoding, while non-text modalities, such as images, video, audio, and actions, are synthesized through iterative denoising. This unified architecture enables Cosmos3 to model heterogeneous modalities within a single framework while preserving generation mechanisms best suited to each modality.

**This model was developed based on:**  [Cosmos Framework](https://github.com/nvidia/cosmos-framework)

**Number of trainable model parameters:**

**Released on: 07/20/2026**
- Cosmos3-Edge: 4B
- Cosmos3-Edge-Policy-DROID: 4B
- Cosmos3-Super-Image2Video-4Step: 64B
- Cosmos3-Super-Text2Image-4Step: 64B

**Released on: 05/31/2026**
- Cosmos3-Nano: 16B
- Cosmos3-Super: 64B
- Cosmos3-Nano-Policy-DROID: 16B
- Cosmos3-Super-Image2Video: 64B
- Cosmos3-Super-Text2Image: 64B

## Input/Output Specifications

- **Generator Input**
  - **Input Type(s)**: Text, Image, Video, Action Trajectory
  - **Input Format(s)**:
    - Text: String
    - Image: jpg, png, jpeg, webp
    - Video: mp4
    - Action: json (1D list)
  - **Input Parameters**:
    - Text: One-dimensional (1D)
    - Image: Two-dimensional (2D)
    - Video: Three-dimensional (3D)
    - Action trajectory: One-dimensional (1D)
  - **Other Properties Related to Input**:
    - For video inputs, we accept various resolutions, including 480p and 256p.
    - Image and video inputs are RGB color (8 bits per channel, sRGB color space); grayscale inputs are not supported.
    - Action input is a per-frame sequence of robot/agent state or control values (e.g., joint positions, gripper state, camera pose). The full input is a 2D array shaped (T, D), where T is the number of frames and D is the embodiment-specific dimensionality listed below.
    - Input action is only supported for the DROID robot platform (8D).
  - **Input Size and Length limits:**
    - **Text:** 4096 tokens
    - **Image:** 256p and 480p resolution at one of these aspect ratios (16:9, 4:3, 1:1, 3:4, 9:16)
    - **Video:** 256p and 480p resolution at one of these aspect ratios (16:9, 4:3, 1:1, 3:4, 9:16). Max number of frames = 5.
    - **Action:** 16 – 400 video frames
- **Generator Output**
  - **Output Type(s)**: Image, video, action, text
  - **Output Format(s)**:
    - Image: JPG
    - Video: MP4
    - Action: 1D list (.json)
    - Text: string
  - **Output Parameters**:
    - Image: Two-dimensional (2D)
    - Video: Three-dimensional (3D)
    - Action: One-dimensional (1D)
    - Text: One-dimensional (1D)
  - **Other Properties Related to Output**:
    - The generated video is an MP4 file, with the resolution, frame rate, and duration specified in the input.
    - Video generation supports durations from 5 to 400 frames, with 189 frames as the default generation duration.
    - The generated action is only supported for the DROID robot platform (8D).
    - Video: mp4 at the FPS specified in input
    - Image: JPEG
- **Reasoner Input**
  - **Input Type(s)**: Text, Text+Image, Text+Video
  - **Input Format(s)**:
    - Text: String
    - Image: jpg, png, jpeg, webp
    - Video: mp4
  - **Input Parameters**:
    - Text: One-dimensional (1D)
    - Image: Two-dimensional (2D)
    - Video: Three-dimensional (3D)
  - **Other Properties Related to Input**:
    - Video inputs are recommended at a frame rate of 4 fps.
    - Long-context inputs supported up to 256K tokens.
  - **Input Size and Length limits:**
    - **Text:** Up to 256K tokens (context window).
    - **Image:** Standard input image formats; passed as file or URL.
    - **Video:** mp4 at the recommended 4 fps.
- **Reasoner Output**
  - **Output Type(s)**: Text
  - **Output Format(s)**:
    - Text: string
  - **Output Parameters**:
    - Text: One-dimensional (1D)
  - **Other Properties Related to Output**:
    - Default `max_tokens=4096+` is recommended for reasoning outputs; longer outputs may be requested.
    - Reasoning outputs may include structured chain-of-thought, 2D/3D point localization, and bounding-box coordinates for vision-based tasks.

The video content visualizes the input text description as a short animated scene, capturing key elements within the specified time constraints.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g., GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

**Runtime Engine(s):**

- [PyTorch](https://github.com/nvidia/cosmos3)

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Hopper

**Operating System(s):**

- Linux (We have not tested on other operating systems.)

**Note:** Only BF16 precision is tested. Other precisions like FP4, FP8, and FP16 are not officially supported.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Training, Testing, and Evaluation Datasets

### Dataset Overview

- **Total Size:** 1.3B data points
- **Total Number of Datasets:** 393 dataset entries
- **Dataset partition:** Training [100%], Testing [N/A — evaluation benchmarks used separately], Validation [N/A — evaluation benchmarks used separately]
- **Time period for training data collection:** 2024–2026
- **Time period for testing data collection:** N/A (standard public benchmarks)
- **Time period for validation data collection:** N/A (standard public benchmarks)

Raw data from internal and external sources is transformed into training-ready data through multiple stages of curation, filtering, and quality review. Data acquisition spans diverse multimodal sources — robotics, autonomous driving, industrial environments, indoor and outdoor scenes, varied lighting and weather conditions, camera viewpoints, object categories, and human activities — to broaden coverage across Physical AI operating environments. Automated filtering pipelines remove corrupted, duplicate, low-quality, and restricted content. Metadata analysis, heuristic rules, and model-assisted classifiers are applied during preprocessing to flag anomalous distributions and low-diversity subsets. Human review supplements automated filtering for selected datasets, benchmark construction, and targeted quality analysis. Datasets are balanced across modalities and task categories — visual reasoning, text-to-image, text-to-video, image-to-video, audio generation, video transfer, action-conditioned generation, and action command generation — to reduce overrepresentation of narrow domains. Synthetic and simulation-based augmentation supplements coverage of rare physical interactions and edge-case scenarios. Deduplication and provenance tracking are applied across the corpus. The resulting processed data is converted into model-ready tokenized or encoded representations through modality-specific preprocessors before training begins.

Training datasets passed through multiple layers of automated and manual safeguards designed to reduce the presence of harmful or policy-violating content across categories including weapons and weapons-related instructional content, criminal planning, child sexual abuse material (CSAM), non-consensual intimate imagery (NCII), sexual content involving minors, harassment, hate speech, profanity, threats and incitement to violence, self-harm or suicide-related content, and graphic violence. Data sources are reviewed for licensing compatibility, provenance, and alignment with internal data governance and safety policies before admission into training corpora. Automated filtering pipelines combine multiple detection strategies: hash-matching against known CSAM and NCII reference databases; classifier-based moderation models trained for explicit sexual content, hate speech, violence, weapons imagery, and other restricted categories; keyword and regex-based screening for criminal-planning, threats, and self-harm phrases in text data; metadata and provenance heuristics for source-level risk signals; and embedding-based anomaly detection to surface samples that fall outside expected distributions. Human review and targeted audits supplement automated filtering for selected datasets, benchmark construction, and safety-sensitive evaluation. For multimodal Physical AI data (robotics, autonomous driving, industrial scenes), additional filtering targets invalid action trajectories, physically implausible interactions, and unsafe control sequences. Synthetic and simulation-generated data are evaluated through internal validation before inclusion. Benchmark evaluations and red-team testing are applied post-training to surface remaining safety gaps across world generation, reasoning, audio, and action tasks. No large-scale data-filtering process can guarantee complete removal of all harmful content; residual risks may remain, particularly in rare edge cases or open-world deployment settings. Ongoing monitoring and dataset review continue post-release.

**Data Modality and Training Data Size**

| Modality | Reasoning Data Sample Count | Generation Data Sample Count |
| -------- | ------------------- | -------------------- |
| Text     | 22M                 | Not Applicable       |
| Image    | 19M                 | 767M                 |
| Video    | 1M                  | 348M                 |
| Audio    | Not Applicable      | 139M                 |
| Action   | Not Applicable      | 7M                   |

**Data Collection Method by dataset**

- Hybrid: Automatic/Sensors, Synthetic, Automated

**Labeling Method by dataset**

- Hybrid: Human, Automated

**Properties:** The training, testing, and evaluation datasets consist of diverse multimodal video, image, audio, action, synthetic, and sensor-conditioned data sourced from NVIDIA-owned data and publicly available, commercially permissive datasets. These datasets are curated to exclude known restricted content and to support building an Omni model that learns to generate and reason about dynamic physical environments across world reasoning and generation tasks.

### Public Datasets

| Dataset                   | Samples     |
|---------------------------|-------------|
| OpenImage                 | 1.2M        |
| Coyo700M                  | 100M        |
| YouTube Video             | 340M        |
| UMI                       | 4.5M        |

### Private Datasets

| Dataset                   | Samples     |
|---------------------------|-------------|
| Egocentric                | 7M          |
| Nexar                     | 0.6M        |
| AgiBot                    | 0.2M        |
| HOI                       | 0.3M        |

### Synthetic Datasets

| Dataset                                 | Samples     |
|-----------------------------------------|-------------|
| synthetic images generated using HiDream-I1       | 15M         |
| synthetic images generated using Qwen-Image-2512  | 14M         |
| synthetic captions generated using Qwen3-VL       | 1115M       |

## Evaluation Datasets

**Data Collection Method by dataset**

- Hybrid: Automatic/Sensors, Synthetic, Automated

**Labeling Method by dataset**

- Hybrid: Human, Automated

**Properties:** The training, testing, and evaluation datasets consist of diverse multimodal video, image, audio, action, synthetic, and sensor-conditioned data sourced from NVIDIA-owned data and publicly available, commercially permissive datasets. These datasets are curated to exclude known restricted content and to support building an Omni model that learns to generate and reason about dynamic physical environments across world reasoning and generation tasks.

## Benchmarks

### Action

#### RoboLab

![RoboLab — task success rates across language specificity and difficulty levels](images/benchmark-robolab.png)

### PBR (Performance Benchmark Reporting)


- Single-GPU end-to-end inference latency (seconds) for Cosmos3-Edge-Policy-DROID.
- Policy DROID generates an action chunk with shape `[16, 8]`: 16 future action timesteps with 8 action values per timestep.
- The canonical policy request uses one input image, `num_frames=17`, `fps=5`, and `action_chunk_size=16`.
- The 17 generated frames represent the rollout associated with the action chunk; they are not a 17-frame input video.
- An empty cell means not measured, not unsupported.

| Platform | Memory | Configuration | vLLM-Omni E2E Latency (s) | PyTorch E2E Latency (s) |
|---|---:|---|---:|---:|
| NVIDIA B200 SXM | 192 GB | Default | 0.99 | — |
| NVIDIA H100 SXM | 80 GB | Default | 1.41 | 1.25 |
| NVIDIA H100 NVL | 96 GB | Default | 1.37 | 1.28 |
| NVIDIA H20 SXM | 96 GB | Default | 3.41 | 2.92 |
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB | Default | 1.87 | 1.65 |
| NVIDIA DGX Station | 128 GB unified memory | Default | 8.11 | 1.77 |
| NVIDIA DGX Spark | 128 GB unified memory | Default | 7.66 | 6.56 |
| NVIDIA Jetson AGX Thor T5000 | 128 GB | MAXN | 6.32 | — |
| NVIDIA Jetson T3000 | 32 GB | 1100 MHz | 8.63 | — |
| NVIDIA Jetson T2000 | 16 GB | 702 MHz, `THOR_NANO` | — | — |

#### Real-time vLLM-Omni Policy DROID with `[32, 8]` action chunks

These measurements use 320 × 192 observations, 33 generated rollout frames, 30 denoising steps, guidance 1.0, and a 5 Hz control rate. For a 32-action chunk, the real-time budget at 5 Hz is 6.4 seconds. End-to-end (E2E) latency includes client-side HTTP overhead, while inference latency is reported by the server. Each value follows one discarded warmup and two timed runs.

| Platform | E2E Latency | Inference Latency | RTF at 5 Hz | Real-time at 5 Hz |
|---|---:|---:|---:|---|
| NVIDIA Jetson AGX Thor T5000, 128 GB, MAXN | 2.59 s | 2.43 s | 2.47 | Yes |
| NVIDIA Jetson AGX Thor T4000, 64 GB, MAXN, 1530 MHz | 3.24 s | 3.14 s | 1.98 | Yes |
| NVIDIA Jetson Thor T3000, 32 GB, 1100 MHz (emulated) | 4.05 s | 3.84 s | 1.58 | Yes |
| NVIDIA Jetson Thor T2000, 16 GB, 765 MHz, `THOR_NANO` | 7.63 s | 7.38 s | 0.84 | No |

#### Real-time PyTorch Policy DROID with `[32, 8]` action chunks

These measurements use an action chunk of `[32, 8]`: 32 future action timesteps with 8 action values per timestep. They are reported separately from the `[16, 8]` Policy DROID measurements above because the action chunk, input processing, and inference configuration differ.

Real-time factor (RTF) is the available time per action chunk divided by measured end-to-end latency; values greater than or equal to 1.0 meet the corresponding real-time budget. For a 32-action chunk, the budgets are 6.4 seconds at 5 Hz and 2.133 seconds at 15 Hz.

These loopback measurements use the PyTorch Policy DROID implementation with one request at a time over a persistent connection, a 640 × 540 observation mapped to a 544 × 736 processing bucket, four UniPC denoising steps, guidance 3.0, and `conditioning_fps=15`. The table evaluates each measured latency against both the 5 Hz and 15 Hz budgets; these are not separate timing runs at the two control rates.

| Platform | Median E2E | Observed Range | Timed Requests | RTF at 5 Hz | RTF at 15 Hz | Real-time at 5 / 15 Hz |
|---|---:|---:|---:|---:|---:|---|
| NVIDIA Jetson AGX Thor T5000, 128 GB, MAXN, 1575 MHz | 1.528 s | 1.503–1.673 s | 16 | 4.19 | 1.40 | Yes / Yes |
| NVIDIA Jetson AGX Thor T4000, 64 GB-class, MAXN, 1530 MHz | 2.208 s | 2.148–2.425 s | 9 | 2.90 | 0.97 | Yes / No |
| NVIDIA Jetson Thor T3000, 32 GB, 1100 MHz (emulated) | 2.632 s | 2.595–2.687 s | 4 | 2.43 | 0.81 | Yes / No |
| NVIDIA Jetson Thor T2000, 16 GB, 765 MHz, `THOR_NANO` | 5.195 s | 5.176–5.224 s | 9 | 1.23 | 0.41 | Yes / No |

The T4000 median is approximately 3.5% above the 15 Hz budget. Its fastest observed request was 2.148 seconds, approximately 0.7% above the budget.

The real-time vLLM-Omni and PyTorch measurements should not be compared directly because they use different input processing, denoising configurations, runtime stacks, and measurement protocols. The T3000 results were measured on an emulated configuration with a 32 GB memory limit, 12 SMs, a 1100 MHz GPU clock cap, and CPU caps. DRAM bandwidth was not capped, so a production T3000 may be slower. The T2000 PyTorch measurement used expandable-segment memory allocation and 2.6 GB of swap, with other Cosmos containers stopped; no swap-out was observed during the audited inference calls.

## Usage

- See [Cosmos](https://github.com/nvidia/cosmos) for details.

### Quickstart

Cosmos3-Edge-Policy-DROID is served by a policy **Server** that streams actions to a **Client** driving a simulated or real robot. This example uses [`RoboLab`](https://github.com/NVlabs/RoboLab), a simulation benchmark for task-generalist policies, as the client. Start the server first, then connect the client.

#### Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```bash
docker build \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```bash
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>
docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

Inside the container, start the policy server:

```
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint_path nvidia/Cosmos3-Edge-Policy-DROID \
  --port 8000 \
  --format-prompt-as-json True
```

#### Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```bash
./docker/build_docker.sh latest
```

Launch the container:

```bash
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

To evaluate across multiple sub-environments in parallel in headless mode:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```

Example output:

<video controls width="864" height="480" src="
https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID/resolve/main/assets/Pick_up_the_banana_and_place_it_in_the_bowl_0_env0_viewport.mp4"></video>

## Limitations

Cosmos3 may produce imperfect outputs in challenging scenarios. Generation artifacts include temporal inconsistency, unstable camera or object motion, imprecise physical interactions, inaccurate audio-video synchronization, and action-state drift — especially in long-horizon or high-resolution outputs. Reasoning may also be incorrect: object states, causal relationships, spatial geometry, temporal ordering, agent intent, and future outcomes can be misinferred, and complex or long-context inputs may yield hallucinated entities, inconsistent interpretations, or implausible predictions. Because the model lacks an explicit physics simulator, 3D geometry, 4D space-time evolution, object permanence, contact dynamics, and physical laws are only approximated — producing artifacts such as disappearing or morphing objects, unrealistic collisions, and physically implausible motions. Quality further degrades in out-of-distribution environments, safety-critical edge cases, and domains underrepresented in training.

Cosmos3 outputs should not be treated as physically accurate simulation, reliable ground-truth reasoning, or safety-certified decision making. Applications involving robotics control, autonomous systems, scientific simulation, or safety-critical planning require additional validation, external constraints, system-level safety analysis, and domain-specific guardrails before deployment.

## Inference

**Acceleration Engine:** [PyTorch](https://pytorch.org/)

**Test Hardware:** B200, H100, H20, RTX PRO 6000, DGX Station, DGX Spark, Jetson Thor

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](EXPLAINABILITY.md), [Bias](BIAS.md), [Safety & Security](SAFETY.md), and [Privacy](PRIVACY.md) subcards. Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
