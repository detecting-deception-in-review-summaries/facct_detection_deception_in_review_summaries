"""Deploy the pinned Qwen judge and all GPU-backed scoring metrics on Modal."""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import modal

CONFIG_ENV_NAME = "DECEPTION_MODAL_CONFIG"

if os.environ.get("MODAL_IS_REMOTE") == "1":
    EXPERIMENT_CONFIG = json.loads(os.environ[CONFIG_ENV_NAME])
    JUDGE_NAME = EXPERIMENT_CONFIG["geval_judge_name"]
    JUDGE_CONFIG = EXPERIMENT_CONFIG["geval_judge_model"]
else:
    from pipeline.common import (
        active_geval_judge,
        geval_judge_app_name,
        load_config,
    )

    EXPERIMENT_CONFIG = load_config()
    # DECEPTION_JUDGE deploys a judge other than the configured one without
    # editing the file, which is what makes `geval_judge` a one-line switch
    # rather than an edit-deploy-revert cycle.
    JUDGE_NAME, JUDGE_CONFIG = active_geval_judge(
        EXPERIMENT_CONFIG,
        os.environ.get("DECEPTION_JUDGE"),
    )

MODAL_CONFIG = EXPERIMENT_CONFIG["modal"]
QWEN_CONFIG = EXPERIMENT_CONFIG["models"]["qwen"]

# The generation Qwen and the G-Eval judge are separate entries now, so the
# vLLM server keeps serving `models.qwen` while the scorer loads whichever
# judge is active.
if os.environ.get("MODAL_IS_REMOTE") == "1":
    APP_NAME = EXPERIMENT_CONFIG["geval_judge_app_name"]
else:
    APP_NAME = geval_judge_app_name(EXPERIMENT_CONFIG, JUDGE_CONFIG)
JUDGE_CHAT_TEMPLATE_KWARGS = dict(
    JUDGE_CONFIG.get("chat_template_kwargs") or {}
)
# FlashAttention-2 by default. A judge whose heads exceed its 256-dimension
# limit (Gemma 4's global layers are 512) names `sdpa` instead, and the scorer
# then pins PyTorch to the memory-efficient kernel: the alternative it would
# otherwise fall back to materialises every attention matrix, which at this
# experiment's longest prompts does not fit beside the weights.
JUDGE_ATTENTION = str(
    JUDGE_CONFIG.get("attn_implementation") or "flash_attention_2"
)

if JUDGE_ATTENTION not in {"flash_attention_2", "sdpa"}:
    raise ValueError(
        f"Judge {JUDGE_NAME} sets attn_implementation={JUDGE_ATTENTION!r}; "
        "the G-Eval scorer supports 'flash_attention_2' and 'sdpa'."
    )
# Where the score is read. By default it is the first token of the judge's
# answer, which is where Qwen and OLMo write their digit. A judge that first
# restates the prompt's closing form line - Gemma 4 answers "- Consistency: 5",
# with 5e-09 on a digit at the first position - sets `answer_prefix:
# form_line`, and its answer is started with that line so the score is read at
# the position it actually writes the digit: directly after "- Consistency: ",
# as in the original completion-style G-Eval.
JUDGE_ANSWER_PREFIX = JUDGE_CONFIG.get("answer_prefix")
# A judge may name its own GPU; the rest share `modal.qwen_gpu`. The card can
# move a score - see geval_judges.gemma4 in experiment.yaml - so a judge that
# sets one carries it in its revision, and must never mix two in one run.
JUDGE_GPU = str(JUDGE_CONFIG.get("gpu") or MODAL_CONFIG["qwen_gpu"])

if JUDGE_ANSWER_PREFIX not in {None, "form_line"}:
    raise ValueError(
        f"Judge {JUDGE_NAME} sets answer_prefix={JUDGE_ANSWER_PREFIX!r}; "
        "the G-Eval scorer supports 'form_line' or no prefix."
    )
QWEN_GPU = MODAL_CONFIG["qwen_gpu"]
QWEN_CPU_CORES = float(MODAL_CONFIG["qwen_cpu_cores"])
QWEN_SCORE_BATCH_SIZE = int(MODAL_CONFIG["qwen_score_batch_size"])
QWEN_SCORE_MAX_INPUT_TOKENS = int(
    MODAL_CONFIG["qwen_score_max_input_tokens"]
)
QWEN_SCORE_TORCH_VERSION = str(
    MODAL_CONFIG["qwen_score_torch_version"]
)
QWEN_SCORE_TRANSFORMERS_VERSION = str(
    MODAL_CONFIG["qwen_score_transformers_version"]
)
QWEN_SCORE_FLASH_ATTENTION_VERSION = str(
    MODAL_CONFIG["qwen_score_flash_attention_version"]
)
QWEN_SCORE_BACKEND_VERSION = str(
    MODAL_CONFIG["qwen_score_backend_version"]
)
QWEN_SCORE_PREFIX_CACHE = bool(
    MODAL_CONFIG.get("qwen_score_prefix_cache", False)
)
QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS = int(
    MODAL_CONFIG.get("qwen_score_prefix_cache_min_tokens", 512)
)
QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS = int(
    MODAL_CONFIG.get("qwen_score_prefix_cache_max_tokens", 32000)
)
QWEN_SCORE_PREFIX_CACHE_TOLERANCE = float(
    MODAL_CONFIG.get("qwen_score_prefix_cache_tolerance", 0.005)
)
QWEN_SCORE_FLASH_LINEAR_ATTENTION_VERSION = str(
    MODAL_CONFIG["qwen_score_flash_linear_attention_version"]
)
QWEN_SCORE_CAUSAL_CONV1D_VERSION = str(
    MODAL_CONFIG["qwen_score_causal_conv1d_version"]
)
QWEN_SCORE_CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    f"v{QWEN_SCORE_CAUSAL_CONV1D_VERSION}/"
    f"causal_conv1d-{QWEN_SCORE_CAUSAL_CONV1D_VERSION}+cu12torch2.9cxx11abiTRUE-"
    "cp312-cp312-linux_x86_64.whl"
)
QWEN_SCORE_FLASH_ATTENTION_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-"
    "cp312-cp312-linux_x86_64.whl"
)
METRIC_GPU = MODAL_CONFIG["metric_gpu"]
LEGACY_METRIC_GPU = MODAL_CONFIG["legacy_metric_gpu"]
FACTGRAPH_GPU = MODAL_CONFIG["factgraph_gpu"]
QWEN_SCALEDOWN_WINDOW = int(
    MODAL_CONFIG["qwen_scaledown_window_seconds"]
)
METRIC_SCALEDOWN_WINDOW = int(
    MODAL_CONFIG["scaledown_window_seconds"]
)

MODEL_NAME = QWEN_CONFIG["model"]
# Two different jobs, deliberately separated: the download ref is optional and
# may be None (track the repo default), while the identity string is what
# results are stamped with and must always exist.
MODEL_REVISION = QWEN_CONFIG.get("revision")
MODEL_IDENTITY = MODEL_REVISION or MODEL_NAME
MODEL_MAX_TOKENS = int(QWEN_CONFIG["context_tokens"])

# The judge the G-Eval scorer loads, which is no longer required to be the
# model the vLLM server generates with.
JUDGE_MODEL_NAME = JUDGE_CONFIG["model"]
JUDGE_MODEL_REVISION = JUDGE_CONFIG.get("revision")
JUDGE_MODEL_IDENTITY = JUDGE_MODEL_REVISION or JUDGE_MODEL_NAME
JUDGE_MAX_TOKENS = int(JUDGE_CONFIG["context_tokens"])

if QWEN_SCORE_MAX_INPUT_TOKENS > JUDGE_MAX_TOKENS:
    raise ValueError(
        f"qwen_score_max_input_tokens is {QWEN_SCORE_MAX_INPUT_TOKENS}, which "
        f"exceeds the {JUDGE_MAX_TOKENS}-token window configured for judge "
        f"{JUDGE_NAME} ({JUDGE_MODEL_NAME}). Scoring may never accept a prompt "
        "the judge cannot hold."
    )

if QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS > QWEN_SCORE_MAX_INPUT_TOKENS:
    raise ValueError(
        "qwen_score_prefix_cache_max_tokens is "
        f"{QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS}, above the "
        f"{QWEN_SCORE_MAX_INPUT_TOKENS}-token scoring limit. A prompt that "
        "long is rejected before the cache is ever consulted."
    )

if QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS >= QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS:
    raise ValueError(
        "qwen_score_prefix_cache_min_tokens "
        f"({QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS}) must be below "
        f"qwen_score_prefix_cache_max_tokens "
        f"({QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS}), or no prompt is ever both "
        "long enough to cache and short enough to be allowed one."
    )
EXPERIMENT_SEED = int(EXPERIMENT_CONFIG["seed"])
# The judge is the first component, so a score names the model that produced
# it and two judges never share a score identity.
QWEN_SCORE_REVISION = ";".join(
    (
        JUDGE_MODEL_IDENTITY,
        f"torch=={QWEN_SCORE_TORCH_VERSION}",
        f"transformers=={QWEN_SCORE_TRANSFORMERS_VERSION}",
        f"flash-attn=={QWEN_SCORE_FLASH_ATTENTION_VERSION}",
        f"fla=={QWEN_SCORE_FLASH_LINEAR_ATTENTION_VERSION}",
        f"causal-conv1d=={QWEN_SCORE_CAUSAL_CONV1D_VERSION}",
        QWEN_SCORE_BACKEND_VERSION,
        # Only for a judge that sets it, so every other revision is unchanged.
        *(
            (f"answer-prefix={JUDGE_ANSWER_PREFIX}",)
            if JUDGE_ANSWER_PREFIX
            else ()
        ),
        *((f"gpu={JUDGE_CONFIG['gpu']}",) if JUDGE_CONFIG.get("gpu") else ()),
    )
)

# Modal writes this into a Dockerfile ENV line inside single quotes, so a quote
# in any value ends the string and the build fails to parse. JSON can carry the
# character as an escape instead, which reads back identically.
REMOTE_CONFIG_ENV = {
    CONFIG_ENV_NAME: json.dumps(
        {
            "seed": EXPERIMENT_SEED,
            "modal": MODAL_CONFIG,
            "models": {
                "qwen": QWEN_CONFIG,
            },
            "geval_judge_name": JUDGE_NAME,
            "geval_judge_model": JUDGE_CONFIG,
            "geval_judge_app_name": APP_NAME,
            # The metric containers construct their backends from
            # `metrics.*.options`, so the block has to travel with them. It is
            # baked into the image environment, which means editing an option
            # invalidates the layer and the next deploy rebuilds it - that is
            # the intent: the options are part of what a score means.
            # `report` only describes a metric for the paper's settings table,
            # so it stays out: rewording it must not rebuild any image.
            "metrics": {
                name: {key: value for key, value in specification.items() if key != "report"}
                for name, specification in EXPERIMENT_CONFIG["metrics"].items()
            },
        }
    ).replace("'", "\\u0027")
}

GPU_COUNT = 1
MODEL_PORT = 8000
MINUTE = 60
QWEN_READINESS_TIMEOUT = 25 * MINUTE
MODEL_ROOT = Path("/models")
SCORING_REMOTE_PATH = "/root/scoring.py"
SCORING_LOCAL_PATH = Path(__file__).parent / "scoring.py"

SUMMAC_MODEL_NAME = "tals/albert-xlarge-vitaminc-mnli"
SUMMAC_MODEL_REVISION = "3082ba54344bd9ddada2be1c5e9b4131721d2a5d"
BARTSCORE_MODEL_NAME = "facebook/bart-large-cnn"
BARTSCORE_MODEL_REVISION = "37f520fa929c961707657b28798b30c003dd100b"
SUPERT_MODEL_NAME = "sentence-transformers/bert-large-nli-stsb-mean-tokens"
SUPERT_MODEL_REVISION = "4088dbef2070e7bf5a18e9881698709d2b6f4050"
SUPERT_SENTENCE_TRANSFORMERS_VERSION = "3.4.1"

# The base checkpoint QAFactEval's question generator is built on. Its AllenNLP
# archive stores the name rather than the weights, so loading it calls
# `BartForConditionalGeneration.from_pretrained("facebook/bart-large")` and only
# then loads the archive's own `weights.th` over the result. That call is
# pinned and vendored here rather than left to the Hub because transformers
# 3.0.2 splits its endpoints: configs and vocabularies come from the legacy
# `s3.amazonaws.com/models.huggingface.co` bucket, which still answers, and
# weights from `cdn.huggingface.co`, which no longer resolves. See
# `redirect_pinned_hub_models` in scoring.py.
QAFACTEVAL_BASE_MODEL_NAME = "facebook/bart-large"
QAFACTEVAL_BASE_MODEL_REVISION = "cb48c1365bd826bd521f650dc2e0940aee54720c"

# AlignScore and MiniCheck, code and weights pinned to the commits their
# revisions in scoring.py name.
ALIGNSCORE_COMMIT = "a0936d5afee642a46b22f6c02a163478447aa493"
ALIGNSCORE_CHECKPOINT_REPO = "yzha/AlignScore"
ALIGNSCORE_CHECKPOINT_REVISION = "8509e78d25bb914939fc585c626500c9b2944249"
ALIGNSCORE_BASE_MODEL_NAME = "roberta-large"
ALIGNSCORE_BASE_MODEL_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
MINICHECK_COMMIT = "b58b9fa69acbd1015ec970fa65dd752413a053d2"
MINICHECK_MODEL_NAME = "lytang/MiniCheck-Flan-T5-Large"
MINICHECK_MODEL_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
BESPOKE_MINICHECK_MODEL_NAME = "bespokelabs/Bespoke-MiniCheck-7B"
BESPOKE_MINICHECK_MODEL_REVISION = "1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8"

# FactGraph: the authors' code, their released checkpoint and the AMR parser
# their evaluation script downloads, each pinned to what FactGraphBackend's
# revision names. The two archives are served from outside the Hub, so they are
# pinned by SHA-256 instead of by commit and refused if they ever change.
FACTGRAPH_COMMIT = "165285a483ebf2fc4ffe2202c65349b99dea9626"
FAST_ALIGN_COMMIT = "cab1e9aac8d3bb02ff5ae58218d8d225a039fa11"
AMR_UTILS_COMMIT = "be5534db1312dc7c6ba25ee50eafeb0d0f5e3f69"
FACTGRAPH_ARCHIVES = (
    (
        "https://public.ukp.informatik.tu-darmstadt.de/ribeiro/factgraph/factgraph.tar.gz",
        "e07e209a2af0e20f7e3fcc4ac8fb847a8e0e3bd26125576ee0ac6831e1daa1e2",
        "checkpoints",
        "factgraph",
    ),
    (
        "https://github.com/bjascob/amrlib-models/releases/download/"
        "parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz",
        "2fcfe2113527accb752b6c574994022b3375b66d59da5042356c0f853e879849",
        "",
        "model_parse_xfm_bart_large-v0_1_0",
    ),
    # The sentence encoder as sentence-transformers 1.2.0 fetches it by name,
    # saved by that same version. The Hub's copy is in the 2.x layout, which
    # 1.2.0 cannot read (its root config.json is not the version stamp 1.2.0
    # looks for).
    (
        "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/"
        "v0.2/paraphrase-mpnet-base-v2.zip",
        "a52c823fc2fe3d6bf544bf311d2e274e2d46b3075963bffd5d7f4f8b74f8a41f",
        "paraphrase-mpnet-base-v2",
        "0_Transformer",
    ),
)
FACTGRAPH_ELECTRA_NAME = "google/electra-base-discriminator"
FACTGRAPH_ELECTRA_REVISION = "1ae76a97c7e84a4e640876a07453fccd636f0667"

SUMMAC_MODEL_PATH = MODEL_ROOT / "summac-vitc"
BARTSCORE_MODEL_PATH = MODEL_ROOT / "bartscore"
SUPERT_MODEL_PATH = MODEL_ROOT / "supert-sbert"
QAFACTEVAL_MODEL_PATH = MODEL_ROOT / "qafacteval"
QAFACTEVAL_BASE_MODEL_PATH = QAFACTEVAL_MODEL_PATH / "bart-large"
SUMMAC_CONVOLUTION_PATH = MODEL_ROOT / "summac_conv_vitc_sent_perc_e.bin"
ALIGNSCORE_CHECKPOINT_PATH = MODEL_ROOT / "alignscore" / "AlignScore-large.ckpt"
FACTGRAPH_MODEL_PATH = MODEL_ROOT / "factgraph"
# Both libraries load their models by Hub name, not by path, so those models
# live in a Hub-layout cache that the two images read offline.
HUB_MODEL_CACHE = MODEL_ROOT / "hub"

QAFACTEVAL_DOWNLOADS = {
    "generation/model.tar.gz": "1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU",
    "answering/model.zip": "1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8",
    "lerc/model.tar.gz": "193K7v6pjOtuXdlMenQW-RzF6ft-xY2qd",
    "lerc/pretraining.tar.gz": "1fWBahDT-O1mpsbND300cuZuF73mfObzH",
}

app = modal.App(APP_NAME)

hugging_face_cache = modal.Volume.from_name(
    MODAL_CONFIG["hugging_face_cache_volume"],
    create_if_missing=True,
)
vllm_cache = modal.Volume.from_name(
    MODAL_CONFIG["vllm_cache_volume"],
    create_if_missing=True,
)
model_weights = modal.Volume.from_name(
    MODAL_CONFIG["model_volume"],
    create_if_missing=True,
)
read_only_model_weights = model_weights.with_mount_options(
    read_only=True,
)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates")
    .uv_pip_install(
        "gdown==5.2.0",
        "huggingface-hub==0.28.0",
    )
    .env(REMOTE_CONFIG_ENV)
)


@app.function(
    image=download_image,
    timeout=2 * 60 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hugging_face_cache,
        str(MODEL_ROOT): model_weights,
    },
)
def prepare_model_weights() -> dict[str, str]:
    """Download every pinned checkpoint once and commit both Modal Volumes."""

    import shutil
    import tarfile
    import urllib.request
    import zipfile

    import gdown
    from huggingface_hub import hf_hub_download, snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # Generation, modification, and scoring all refuse an unexpanded revision.
    # This is the one place that downloads weights, and without the same check
    # the Hub reports the placeholder as a missing snapshot, which reads like a
    # network fault rather than an unpinned checkpoint.
    if MODEL_REVISION and "$" in MODEL_REVISION:
        raise ValueError(
            f"models.qwen.revision is not pinned: {MODEL_REVISION!r}. "
            "Set it to a Hugging Face commit, or remove the key to track the "
            "repository default."
        )

    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
    )

    # The G-Eval judge may be a different checkpoint from the generation Qwen,
    # and every loader runs with local_files_only=True, so its weights have to
    # be in the volume before the scorer starts.
    if (JUDGE_MODEL_NAME, JUDGE_MODEL_REVISION) != (MODEL_NAME, MODEL_REVISION):
        snapshot_download(
            repo_id=JUDGE_MODEL_NAME,
            revision=JUDGE_MODEL_REVISION,
        )

    pinned_hugging_face_models = (
        (
            SUMMAC_MODEL_NAME,
            SUMMAC_MODEL_REVISION,
            SUMMAC_MODEL_PATH,
        ),
        (
            BARTSCORE_MODEL_NAME,
            BARTSCORE_MODEL_REVISION,
            BARTSCORE_MODEL_PATH,
        ),
        (
            SUPERT_MODEL_NAME,
            SUPERT_MODEL_REVISION,
            SUPERT_MODEL_PATH,
        ),
    )

    for model_name, revision, destination in pinned_hugging_face_models:
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_dir=destination,
        )

    hf_hub_download(
        repo_id=ALIGNSCORE_CHECKPOINT_REPO,
        filename=ALIGNSCORE_CHECKPOINT_PATH.name,
        revision=ALIGNSCORE_CHECKPOINT_REVISION,
        local_dir=ALIGNSCORE_CHECKPOINT_PATH.parent,
    )

    # Downloaded at a pinned commit, which leaves no `refs/main` behind, and a
    # name loaded offline resolves through `refs/main`. Pointing it at the
    # pinned commit makes the name load exactly that snapshot.
    for model_name, revision, patterns in (
        (
            ALIGNSCORE_BASE_MODEL_NAME,
            ALIGNSCORE_BASE_MODEL_REVISION,
            ["config.json", "merges.txt", "model.safetensors", "tokenizer.json", "vocab.json"],
        ),
        (
            MINICHECK_MODEL_NAME,
            MINICHECK_MODEL_REVISION,
            ["*.json", "pytorch_model.bin", "spiece.model"],
        ),
        (
            # InternLM2 ships its tokenizer and config as code, hence the .py.
            BESPOKE_MINICHECK_MODEL_NAME,
            BESPOKE_MINICHECK_MODEL_REVISION,
            ["*.json", "*.py", "*.safetensors", "tokenizer.model"],
        ),
    ):
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=HUB_MODEL_CACHE,
            allow_patterns=patterns,
        )
        refs = HUB_MODEL_CACHE / f"models--{model_name.replace('/', '--')}" / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)

    if not SUMMAC_CONVOLUTION_PATH.is_file():
        convolution_url = (
            "https://raw.githubusercontent.com/tingofurro/summac/"
            "525cb081993ad1c872d73a9d5d46ea3c3f73c24f/"
            "summac_conv_vitc_sent_perc_e.bin"
        )
        urllib.request.urlretrieve(
            convolution_url,
            SUMMAC_CONVOLUTION_PATH,
        )

    QAFACTEVAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # `local_dir` rather than the Hub cache: transformers 3.0.2 predates the
    # snapshot layout and only recognises a plain directory. Restricted to the
    # files that stack reads, since the same repository also carries TensorFlow,
    # Flax and Rust copies of the weights it would never open.
    snapshot_download(
        repo_id=QAFACTEVAL_BASE_MODEL_NAME,
        revision=QAFACTEVAL_BASE_MODEL_REVISION,
        local_dir=QAFACTEVAL_BASE_MODEL_PATH,
        allow_patterns=[
            "config.json",
            "merges.txt",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )

    for relative_path, google_drive_id in QAFACTEVAL_DOWNLOADS.items():
        destination = QAFACTEVAL_MODEL_PATH / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not destination.is_file():
            downloaded = gdown.download(
                id=google_drive_id,
                output=str(destination),
                quiet=False,
            )

            if downloaded is None:
                raise RuntimeError(f"QAFactEval download failed for {relative_path}.")

    answering_archive = QAFACTEVAL_MODEL_PATH / "answering" / "model.zip"
    answering_marker = QAFACTEVAL_MODEL_PATH / "answering" / ".extracted"

    if not answering_marker.exists():
        with zipfile.ZipFile(answering_archive) as archive:
            archive.extractall(answering_archive.parent)

        answering_marker.touch()

    quip_directory = QAFACTEVAL_MODEL_PATH / "quip-512-mocha"

    if not quip_directory.is_dir():
        quip_archive = QAFACTEVAL_MODEL_PATH / "quip-512-mocha.tar.gz"
        urllib.request.urlretrieve(
            (
                "https://storage.googleapis.com/sfr-qafacteval-research/"
                "quip-512-mocha.tar.gz"
            ),
            quip_archive,
        )

        with tarfile.open(quip_archive) as archive:
            archive.extractall(QAFACTEVAL_MODEL_PATH)

        quip_archive.unlink()

    shutil.rmtree(
        QAFACTEVAL_MODEL_PATH / "answering" / "__MACOSX",
        ignore_errors=True,
    )

    # FactGraph. Both archives come from outside the Hub, so each is checked
    # against its pinned SHA-256 before it is unpacked; the checkpoint is a
    # pickled model, which loading executes.
    import hashlib

    # An earlier version of this function put the Hub's snapshot of the
    # sentence encoder where its archive now unpacks; 1.2.0 cannot read it.
    stale_encoder = FACTGRAPH_MODEL_PATH / "paraphrase-mpnet-base-v2"
    if stale_encoder.is_dir() and not (stale_encoder / "0_Transformer").is_dir():
        shutil.rmtree(stale_encoder)

    for url, sha256, subdirectory, unpacked in FACTGRAPH_ARCHIVES:
        destination = FACTGRAPH_MODEL_PATH / subdirectory

        if (destination / unpacked).is_dir():
            continue

        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / url.rsplit("/", 1)[-1]
        urllib.request.urlretrieve(url, archive)

        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)

        if digest.hexdigest() != sha256:
            archive.unlink()
            raise RuntimeError(
                f"{url} has SHA-256 {digest.hexdigest()}, not the pinned {sha256}."
            )

        shutil.unpack_archive(str(archive), str(destination))
        archive.unlink()

    # `local_dir`, as for QAFactEval: the transformers FactGraph runs on (4.6.1)
    # predates the Hub cache layout and reads a plain directory.
    for model_name, revision, patterns in (
        (
            FACTGRAPH_ELECTRA_NAME,
            FACTGRAPH_ELECTRA_REVISION,
            ["config.json", "pytorch_model.bin", "tokenizer.json",
             "tokenizer_config.json", "vocab.txt"],
        ),
    ):
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_dir=FACTGRAPH_MODEL_PATH / model_name.rsplit("/", 1)[-1],
            allow_patterns=patterns,
        )

    hugging_face_cache.commit()
    model_weights.commit()

    return {
        "qwen": MODEL_REVISION,
        "summac": SUMMAC_MODEL_REVISION,
        "bartscore": BARTSCORE_MODEL_REVISION,
        "supert": SUPERT_MODEL_REVISION,
        "alignscore": ALIGNSCORE_CHECKPOINT_REVISION,
        "minicheck": MINICHECK_MODEL_REVISION,
        "minicheck_7b": BESPOKE_MINICHECK_MODEL_REVISION,
        "qafacteval": (
            "official-download-models.sh@01177f11+"
            f"{QAFACTEVAL_BASE_MODEL_NAME}@{QAFACTEVAL_BASE_MODEL_REVISION}"
        ),
        "factgraph": ";".join(
            [f"{url.rsplit('/', 1)[-1]}@sha256:{sha256[:8]}"
             for url, sha256, _, _ in FACTGRAPH_ARCHIVES]
            + [f"{FACTGRAPH_ELECTRA_NAME}@{FACTGRAPH_ELECTRA_REVISION[:8]}"]
        ),
    }


qwen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            **REMOTE_CONFIG_ENV,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "OMP_NUM_THREADS": str(int(QWEN_CPU_CORES)),
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)

qwen_score_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "accelerate==1.12.0",
        "ninja==1.13.0",
        "packaging==25.0",
        f"torch=={QWEN_SCORE_TORCH_VERSION}",
        f"transformers=={QWEN_SCORE_TRANSFORMERS_VERSION}",
        "wheel==0.45.1",
    )
    .run_commands(
        f'python -m pip install "{QWEN_SCORE_FLASH_ATTENTION_WHEEL}"',
        # Fused kernels for the linear-attention layers. flash-linear-attention
        # is Triton and installs as a pure-Python wheel; causal-conv1d is CUDA
        # and is taken from its prebuilt wheel so nothing compiles here. Naming
        # the wheel directly also avoids its setup.py guessing a URL for a
        # torch version it never published.
        (
            "python -m pip install --no-build-isolation "
            f"flash-linear-attention=={QWEN_SCORE_FLASH_LINEAR_ATTENTION_VERSION}"
        ),
        f'python -m pip install "{QWEN_SCORE_CAUSAL_CONV1D_WHEEL}"',
    )
    .env(
        {
            **REMOTE_CONFIG_ENV,
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": str(int(QWEN_CPU_CORES)),
            "TOKENIZERS_PARALLELISM": "true",
            # The scorer allocates differently shaped activations for every
            # prompt length, which fragments the caching allocator: Gemma 4
            # failed a 1.5 GiB allocation with 2.67 GiB reserved but unused.
            # Expandable segments let freed blocks be reused at any size. Only
            # where memory comes from changes, never what is computed.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


def wait_for_server_port(
    process: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout_seconds: int,
) -> None:
    """Keep Modal in container startup until the child server is listening."""

    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    next_progress_at = started_at

    while True:
        return_code = process.poll()

        if return_code is not None:
            raise RuntimeError(
                "The vLLM server exited before becoming ready "
                f"with status {return_code}."
            )

        try:
            with socket.create_connection(
                (host, port),
                timeout=1.0,
            ):
                elapsed_seconds = round(time.monotonic() - started_at)
                print(
                    f"Qwen server is accepting connections after "
                    f"{elapsed_seconds} seconds.",
                    flush=True,
                )
                return
        except OSError:
            pass

        now = time.monotonic()

        if now >= deadline:
            process.terminate()
            raise RuntimeError(
                "The vLLM server did not listen on "
                f"{host}:{port} within {timeout_seconds} seconds."
            )

        if now >= next_progress_at:
            elapsed_seconds = round(now - started_at)
            print(
                f"Waiting for Qwen server readiness "
                f"({elapsed_seconds} seconds elapsed).",
                flush=True,
            )
            next_progress_at = now + 30

        time.sleep(1)


@app.server(
    image=qwen_image,
    gpu=f"{QWEN_GPU}:{GPU_COUNT}",
    cpu=QWEN_CPU_CORES,
    port=MODEL_PORT,
    startup_timeout=30 * MINUTE,
    scaledown_window=QWEN_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hugging_face_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class QwenServer:
    """Authenticated OpenAI-compatible vLLM server running entirely on Modal."""

    @modal.enter()
    def start(self) -> None:
        command = [
            "vllm",
            "serve",
            MODEL_NAME,
            *(("--revision", MODEL_REVISION) if MODEL_REVISION else ()),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(MODEL_PORT),
            "--dtype",
            "bfloat16",
            "--language-model-only",
            "--attention-backend",
            "FLASH_ATTN",
            "--no-enforce-eager",
            "--max-model-len",
            str(MODEL_MAX_TOKENS),
            "--tensor-parallel-size",
            str(GPU_COUNT),
            "--max-num-seqs",
            "32",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
            "--safetensors-load-strategy",
            "prefetch",
            "--seed",
            str(EXPERIMENT_SEED),
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--trust-remote-code",
            "--generation-config",
            "vllm",
            "--reasoning-parser",
            "qwen3",
            # Lets a per-request `thinking_token_budget` (llms.py's
            # `_qwen_request`) force the model out of reasoning early instead
            # of spending its whole token budget there. Without this flag the
            # field is accepted by the server but has no effect.
            "--reasoning-config",
            json.dumps(
                {
                    "reasoning_start_str": "<think>",
                    "reasoning_end_str": (
                        "I have reasoned enough about this; I will now give "
                        "the final answer.</think>"
                    ),
                }
            ),
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": False}),
        ]

        print("Starting pinned Qwen server:", " ".join(command))
        self.process = subprocess.Popen(command)
        wait_for_server_port(
            process=self.process,
            host="127.0.0.1",
            port=MODEL_PORT,
            timeout_seconds=QWEN_READINESS_TIMEOUT,
        )

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


@app.cls(
    image=qwen_score_image,
    gpu=JUDGE_GPU,
    cpu=QWEN_CPU_CORES,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hugging_face_cache,
    },
)
class QwenGEvalService:
    """Fast, text-only Qwen scorer adapted from the supplied notebook."""

    @modal.enter()
    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            JUDGE_MODEL_NAME,
            revision=JUDGE_MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if JUDGE_ATTENTION == "sdpa":
            # Process-wide rather than a context manager, so it also holds in
            # whichever thread Modal runs a method on. With the math backend
            # off, a call no fused kernel can take raises instead of silently
            # materialising the attention matrix.
            #
            # Both fused kernels stay on because neither covers every call.
            # The memory-efficient kernel takes the 512-dimension global heads
            # and any explicit mask, which is every layer once a prompt passes
            # Gemma's 1024-token window. Below the window transformers drops the
            # sliding layers' mask and passes `enable_gqa=True`, which only the
            # flash kernel accepts - with flash off as well, the first such
            # prompt failed with "Invalid backend". Flash is ineligible for
            # 512-dimension heads and for masked calls, so enabling it changes
            # nothing for prompts at or above the window.
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)

        self.model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            revision=JUDGE_MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            attn_implementation=JUDGE_ATTENTION,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # Read back from the loaded model, so scores record the attention the
        # judge actually ran with rather than the one that was asked for.
        text_config = getattr(self.model.config, "text_config", None)
        self.attention = getattr(
            text_config or self.model.config,
            "_attn_implementation",
            None,
        )

        if self.attention != JUDGE_ATTENTION:
            raise RuntimeError(
                f"Judge {JUDGE_NAME} loaded with attention {self.attention!r}, "
                f"not the configured {JUDGE_ATTENTION!r}."
            )

        score_token_ids: list[int] = []

        for label in range(1, 6):
            token_ids = self.tokenizer.encode(
                str(label),
                add_special_tokens=False,
            )

            if len(token_ids) != 1:
                raise RuntimeError(
                    f"Score label {label} is not one Qwen token: {token_ids}"
                )

            score_token_ids.append(token_ids[0])

        self.score_token_ids = torch.tensor(
            score_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._prefix_cache: Any = None
        self._prefix_ids: list[int] | None = None
        self._previous_ids: list[int] | None = None
        # The last prefix whose snapshot did not fit on the card; see
        # _score_cached. Kept across _reset_prefix_state on purpose.
        self._failed_prefix_ids: list[int] | None = None

        if QWEN_SCORE_PREFIX_CACHE:
            self._validate_prefix_cache()

    def _format_prompt(self, prompt: str) -> str:
        formatted = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            **JUDGE_CHAT_TEMPLATE_KWARGS,
        )

        if JUDGE_ANSWER_PREFIX == "form_line":
            # "- Consistency: " - the trailing space is its own token for
            # Gemma, and the digit follows it.
            formatted += prompt.rstrip().splitlines()[-1] + " "

        return formatted

    @staticmethod
    def _common_prefix_length(left: list[int], right: list[int]) -> int:
        """Count the tokens two prompts share from the start."""

        length = 0

        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break

            length += 1

        return length

    def _reset_prefix_state(self) -> None:
        """Forget the snapshot and the last prompt, freeing the cache's memory."""

        self._prefix_cache = None
        self._prefix_ids = None
        self._previous_ids = None

    def _snapshot_prefix(self, prefix_ids: list[int]) -> None:
        """Prefill `prefix_ids` alone and keep the resulting cache untouched.

        The snapshot is taken over the prefix by itself rather than by cropping
        the cache of a longer prompt. Qwen's hybrid layers carry a recurrent
        linear-attention state whose value depends on every token processed so
        far and cannot be rewound, so a cropped cache would leave those layers
        holding state for a summary the next prompt never contained. Taking the
        snapshot before any summary is appended keeps it exactly the state the
        prefix alone produces, for every layer type.
        """

        torch = self.torch
        self._prefix_cache = None
        self._prefix_ids = None
        input_ids = torch.tensor(
            [prefix_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )

        self._prefix_cache = outputs.past_key_values
        self._prefix_ids = list(prefix_ids)

    def _copy_prefix_cache(self) -> Any:
        """Return a private copy of the snapshot for one scoring pass.

        The model appends the summary's keys, values and recurrent state to
        whatever cache it is handed, so the snapshot itself is never passed in
        directly - one scored summary would otherwise leave its own state
        behind for the next.
        """

        try:
            return copy.deepcopy(self._prefix_cache)
        except Exception as exception:
            raise RuntimeError(
                "The G-Eval prefix cache could not be copied "
                f"({type(exception).__name__}: {exception}). Set "
                "modal.qwen_score_prefix_cache to false to score without it."
            ) from exception

    def _label_distribution(
        self,
        probability_row: list[float],
    ) -> tuple[float, dict[str, float], int]:
        distribution = {
            str(label): float(probability_row[label - 1])
            for label in range(1, 6)
        }
        weighted_score = sum(
            label * distribution[str(label)]
            for label in range(1, 6)
        )
        greedy_score = max(
            range(1, 6),
            key=lambda label: distribution[str(label)],
        )

        return weighted_score, distribution, greedy_score

    def _label_probabilities(self, final_logits: Any) -> tuple[Any, Any]:
        """Return the 1-5 distribution and the share of the vocabulary it held.

        The distribution is renormalised over the five labels, so it reads the
        same whether the judge put all of its mass on a digit or almost none.
        The share is the full-vocabulary probability of the five labels before
        that renormalisation - how far the judge was answering the form at all.
        """

        torch = self.torch
        score_logits = final_logits.index_select(
            dim=-1,
            index=self.score_token_ids,
        )
        probabilities = torch.softmax(score_logits.float(), dim=-1)
        label_mass = (
            torch.softmax(final_logits.float(), dim=-1)
            .index_select(dim=-1, index=self.score_token_ids)
            .sum(dim=-1)
        )

        return probabilities, label_mass

    def _score_cached(self, prompt: str) -> dict[str, Any]:
        """Score one prompt, reusing a shared prefix wherever one is available.

        A cached prefix is used only when it is a token-for-token prefix of the
        current prompt, so reuse can never splice one product's reviews onto
        another's summary. Within a source group the first prompt establishes
        the boundary, the second snapshots it, and the rest reuse it.
        """

        torch = self.torch
        ids = self.tokenizer(
            self._format_prompt(prompt),
            add_special_tokens=False,
        )["input_ids"]
        total_tokens = len(ids)

        if total_tokens > QWEN_SCORE_MAX_INPUT_TOKENS:
            raise ValueError(
                f"A G-Eval prompt contains {total_tokens} tokens, exceeding "
                f"the configured {QWEN_SCORE_MAX_INPUT_TOKENS}-token scoring "
                "limit. The prompt was not silently truncated."
            )

        if total_tokens > QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS:
            # Long enough that a resident cache would compete with activations
            # for the card, so this one prompt is scored the plain way.
            self._reset_prefix_state()
            self._previous_ids = ids

            return self._score_uncached([{"prompt": prompt}])[0]

        reused_tokens = 0

        if self._prefix_ids is not None and (
            self._common_prefix_length(self._prefix_ids, ids)
            == len(self._prefix_ids)
            < total_tokens
        ):
            reused_tokens = len(self._prefix_ids)
        else:
            # The snapshot belongs to another product or metric. Drop it, then
            # see whether the previous prompt shares enough to snapshot a new
            # one for the group now starting.
            self._prefix_cache = None
            self._prefix_ids = None
            boundary = (
                self._common_prefix_length(self._previous_ids, ids)
                if self._previous_ids is not None
                else 0
            )

            if (
                QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS <= boundary < total_tokens
                and ids[:boundary] != self._failed_prefix_ids
            ):
                snapshot_failed = False

                try:
                    self._snapshot_prefix(ids[:boundary])
                    reused_tokens = boundary
                except torch.cuda.OutOfMemoryError:
                    snapshot_failed = True
                    # A snapshot holds a whole prefix's cache on top of the
                    # activations an uncached pass needs, so on the longest
                    # products it can exceed the card where scoring the prompt
                    # uncached does not. The group is scored uncached instead -
                    # the path the startup check holds the cache to - and the
                    # prefix is remembered so that each further summary of the
                    # product does not repeat the failed prefill first.
                    self._prefix_cache = None
                    self._prefix_ids = None
                    self._failed_prefix_ids = ids[:boundary]

                if snapshot_failed:
                    # Outside the except block, for the reason given in
                    # _score_resilient.
                    self._release_memory()

        self._previous_ids = ids

        if reused_tokens == 0:
            return self._score_uncached([{"prompt": prompt}])[0]

        torch.cuda.synchronize()
        started_at = time.monotonic()

        with torch.inference_mode():
            past_key_values = self._copy_prefix_cache()
            input_ids = torch.tensor(
                [ids[reused_tokens:]],
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones(
                    (1, total_tokens),
                    dtype=torch.long,
                    device=self.device,
                ),
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=torch.arange(
                    reused_tokens,
                    total_tokens,
                    device=self.device,
                ),
                logits_to_keep=1,
                return_dict=True,
            )
            probabilities, label_mass = self._label_probabilities(
                outputs.logits[:, -1, :]
            )

        torch.cuda.synchronize()
        elapsed_seconds = time.monotonic() - started_at
        probability_row = probabilities[0].cpu().tolist()
        label_probability_mass = float(label_mass[0].cpu())
        computed_tokens = total_tokens - reused_tokens
        del outputs, past_key_values
        weighted_score, distribution, greedy_score = self._label_distribution(
            probability_row
        )

        return {
            "score": weighted_score,
            "probabilities": distribution,
            "raw_answer": str(greedy_score),
            "details": {
                "backend": "transformers-final-logits",
                "attention": self.attention,
                "label_probability_mass": label_probability_mass,
                "batch_size": 1,
                "input_tokens": total_tokens,
                "batch_input_tokens": total_tokens,
                "batch_seconds": elapsed_seconds,
                "batch_input_tokens_per_second": (
                    computed_tokens / elapsed_seconds
                    if elapsed_seconds > 0
                    else None
                ),
                "logits_to_keep": 1,
                "prefix_cache_used": True,
                "prefix_tokens": reused_tokens,
                "computed_tokens": computed_tokens,
            },
        }

    def _validate_prefix_cache(self) -> None:
        """Refuse to serve if cached and uncached scoring disagree.

        Reusing a prefix is only sound if a snapshot taken over the prefix
        alone reproduces what one forward pass over the whole prompt computes.
        That holds for attention layers by construction, but this model's
        hybrid linear-attention layers carry recurrent state, and whether the
        installed transformers build threads an injected cache through them
        correctly is a property of that build rather than something this file
        can assume. So it is measured once, here, and the container fails at
        startup rather than emitting scores that quietly differ from the
        uncached backend its revision string names.

        Judges with sliding-window layers (OLMo 3 at 4096 tokens, Gemma 4 at
        1024) keep only the last window of keys in those layers' cache, so the
        shared prefix is sized past the widest window: a shorter one would
        validate a cache that never trims, which is not the cache real
        prompts - median ~3.5k tokens, up to ~25k - are scored against.
        """

        shared = "Review 1:\n" + (
            "The item arrived quickly and works exactly as described. " * 500
        )
        # Summaries long enough to resume over roughly as many tokens as a real
        # one. A short suffix passes on kernel paths a long suffix would fail,
        # so checking with a token or two of summary would prove nothing about
        # the prompts this actually serves.
        summaries = [
            (
                "Reviewers were consistently positive about this product. "
                "Most mention that delivery was faster than expected and that "
                "the item matched its description closely. Several highlight "
                "build quality as the strongest point, noting solid materials "
                "and a finish that has held up over months of daily use. A "
                "few reviewers compare it favourably against more expensive "
                "alternatives they had owned previously. Value for money comes "
                "up repeatedly, with buyers saying they would purchase again "
                "and have recommended it to friends and family. The small "
                "number of criticisms concern packaging rather than the "
                "product itself, and none of the reviewers reported a fault "
                "that affected normal use of the item over time."
            ),
            (
                "Opinion in these reviews is sharply divided and often "
                "negative. A large group of buyers report that the item "
                "arrived later than promised, sometimes by more than a week, "
                "and several describe damage to the outer packaging on "
                "arrival. Complaints about durability dominate the lower "
                "ratings, with multiple reviewers saying components loosened "
                "or stopped working within the first month. Others dispute "
                "the description entirely, arguing that dimensions and "
                "materials differ from what was advertised. Positive reviews "
                "exist but are brief and mention little beyond price. Overall "
                "the reviewers who wrote at length were markedly less "
                "satisfied than the rating average alone would suggest."
            ),
        ]
        prompts = [
            f"{shared}\n\nSummary:\n\n{summary}\n\n- Consistency:"
            for summary in summaries
        ]

        self._reset_prefix_state()
        uncached = [
            self._score_uncached([{"prompt": prompt}])[0] for prompt in prompts
        ]
        self._reset_prefix_state()
        cached = [self._score_cached(prompt) for prompt in prompts]
        self._reset_prefix_state()
        self.torch.cuda.empty_cache()

        if not cached[-1]["details"].get("prefix_cache_used"):
            raise RuntimeError(
                "The G-Eval prefix cache never engaged during startup "
                "validation, so its correctness could not be established. "
                "modal.qwen_score_prefix_cache_min_tokens "
                f"({QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS}) may exceed the shared "
                "prefix of the validation prompts."
            )

        model_config = getattr(self.model.config, "text_config", None) or (
            self.model.config
        )
        sliding_window = getattr(model_config, "sliding_window", None) or 0

        if cached[-1]["details"]["prefix_tokens"] <= sliding_window:
            raise RuntimeError(
                "The validation prefix "
                f"({cached[-1]['details']['prefix_tokens']} tokens) fits inside "
                f"this judge's {sliding_window}-token sliding window, so the "
                "check would pass without ever exercising a trimmed cache."
            )

        largest_difference = max(
            abs(
                plain_result["probabilities"][label]
                - cached_result["probabilities"][label]
            )
            for plain_result, cached_result in zip(uncached, cached)
            for label in "12345"
        )

        if largest_difference > QWEN_SCORE_PREFIX_CACHE_TOLERANCE:
            raise RuntimeError(
                "Prefix-cached G-Eval scoring disagrees with uncached scoring "
                f"by {largest_difference:.6f}, above the configured "
                f"{QWEN_SCORE_PREFIX_CACHE_TOLERANCE} tolerance. The scores "
                "this container would produce are not the scores its revision "
                "string promises. Set modal.qwen_score_prefix_cache to false "
                "to score without the cache."
            )

        print(
            json.dumps(
                {
                    "event": "qwen_geval_prefix_cache_validated",
                    "largest_probability_difference": round(
                        largest_difference, 8
                    ),
                    "tolerance": QWEN_SCORE_PREFIX_CACHE_TOLERANCE,
                    "prefix_tokens": cached[-1]["details"]["prefix_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @modal.method()
    def diagnose_prefix_cache(self) -> dict[str, Any]:
        """Measure which way of resuming from a snapshot reproduces one pass.

        The startup check reports only that cached and uncached scoring differ.
        This reports why: it holds the snapshot fixed and varies how the
        remaining tokens are fed - the mask, the cache positions, and whether
        they arrive together or one at a time - so the disagreement can be
        attributed instead of guessed at.
        """

        torch = self.torch
        shared = "Review 1:\n" + (
            "The item arrived quickly and works exactly as described. " * 80
        )
        prompt = (
            f"{shared}\n\nSummary:\n\nReviewers were positive.\n\n- Consistency:"
        )
        ids = self.tokenizer(
            self._format_prompt(prompt),
            add_special_tokens=False,
        )["input_ids"]
        total = len(ids)

        def probabilities_from(logits: Any) -> list[float]:
            selected = logits[:, -1, :].index_select(
                dim=-1,
                index=self.score_token_ids,
            )

            return torch.softmax(selected.float(), dim=-1)[0].cpu().tolist()

        def uncached() -> list[float]:
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )

                return probabilities_from(outputs.logits)

        baseline = uncached()

        def largest_gap(candidate: list[float]) -> float:
            return max(abs(a - b) for a, b in zip(baseline, candidate))

        def resume(
            boundary: int,
            use_mask: bool,
            use_cache_position: bool,
            one_token_at_a_time: bool,
        ) -> list[float]:
            self._snapshot_prefix(ids[:boundary])

            with torch.inference_mode():
                past = copy.deepcopy(self._prefix_cache)
                steps = (
                    [[token] for token in ids[boundary:]]
                    if one_token_at_a_time
                    else [ids[boundary:]]
                )
                position = boundary
                outputs = None

                for step in steps:
                    input_ids = torch.tensor(
                        [step],
                        dtype=torch.long,
                        device=self.device,
                    )
                    keyword_arguments: dict[str, Any] = {}

                    if use_mask:
                        keyword_arguments["attention_mask"] = torch.ones(
                            (1, position + len(step)),
                            dtype=torch.long,
                            device=self.device,
                        )

                    if use_cache_position:
                        keyword_arguments["cache_position"] = torch.arange(
                            position,
                            position + len(step),
                            device=self.device,
                        )

                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=past,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                        **keyword_arguments,
                    )
                    position += len(step)

                return probabilities_from(outputs.logits)

        cache_type = type(self._prefix_cache).__name__ if self._prefix_cache else None
        variants: list[tuple[str, dict[str, Any]]] = [
            ("mask+position (current)", {"use_mask": True, "use_cache_position": True}),
            ("position only", {"use_mask": False, "use_cache_position": True}),
            ("mask only", {"use_mask": True, "use_cache_position": False}),
            ("neither", {"use_mask": False, "use_cache_position": False}),
            (
                "one token at a time",
                {
                    "use_mask": True,
                    "use_cache_position": True,
                    "one_token_at_a_time": True,
                },
            ),
        ]
        findings: list[dict[str, Any]] = []

        for label, options in variants:
            for suffix_length in (1, 24):
                settings = {
                    "one_token_at_a_time": False,
                    **options,
                }

                try:
                    gap = largest_gap(
                        resume(boundary=total - suffix_length, **settings)
                    )
                    findings.append(
                        {
                            "variant": label,
                            "suffix_tokens": suffix_length,
                            "largest_gap": round(gap, 6),
                            "matches": gap <= QWEN_SCORE_PREFIX_CACHE_TOLERANCE,
                        }
                    )
                except Exception as exception:  # noqa: BLE001
                    findings.append(
                        {
                            "variant": label,
                            "suffix_tokens": suffix_length,
                            "error": f"{type(exception).__name__}: {exception}",
                        }
                    )

        # Every resume variant matched the unpadded forward exactly, so the
        # disagreement the startup check reports has to come from the other
        # side of its comparison: _score_uncached left-pads to a multiple of
        # eight, and the padded and unpadded forms of the same prompt are what
        # is being measured against each other here.
        padded = self._score_uncached([{"prompt": prompt}])[0]
        padded_gap = max(
            abs(baseline[label - 1] - padded["probabilities"][str(label)])
            for label in range(1, 6)
        )
        self._reset_prefix_state()
        torch.cuda.empty_cache()

        return {
            "cache_class": cache_type,
            "total_tokens": total,
            "padded_tokens": (total + 7) // 8 * 8,
            "padding_added": ((total + 7) // 8 * 8) - total,
            "padded_vs_unpadded_gap": round(padded_gap, 6),
            "tolerance": QWEN_SCORE_PREFIX_CACHE_TOLERANCE,
            "findings": findings,
        }

    def _score_once(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if QWEN_SCORE_PREFIX_CACHE and len(requests) == 1:
            return [self._score_cached(requests[0]["prompt"])]

        return self._score_uncached(requests)

    def _score_uncached(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        torch = self.torch
        formatted_prompts = [
            self._format_prompt(request["prompt"])
            for request in requests
        ]
        # Never padded for a single prompt, which is every G-Eval call. Left
        # padding measurably changes this model's answer: one pad token on an
        # 831-token prompt moved a label probability by 0.049, a thousand times
        # the 6e-05 that separates two honest kernel paths. The hybrid layers
        # convolve over the sequence, and a pad token is content to a
        # convolution however the attention mask marks it. `pad_to_multiple_of`
        # made the shift depend on prompt length modulo eight, so it varied
        # per prompt rather than cancelling.
        encoded = self.tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=len(formatted_prompts) > 1,
            truncation=False,
            return_tensors="pt",
        )
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        longest_input = max(input_lengths)

        if longest_input > QWEN_SCORE_MAX_INPUT_TOKENS:
            raise ValueError(
                "A G-Eval prompt contains "
                f"{longest_input} tokens, exceeding the configured "
                f"{QWEN_SCORE_MAX_INPUT_TOKENS}-token scoring limit. "
                "The prompt was not silently truncated."
            )

        model_inputs = {
            name: tensor.to(
                self.device,
                non_blocking=True,
            )
            for name, tensor in encoded.items()
        }
        torch.cuda.synchronize()
        started_at = time.monotonic()

        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            probabilities, label_mass = self._label_probabilities(
                outputs.logits[:, -1, :]
            )

        torch.cuda.synchronize()
        elapsed_seconds = time.monotonic() - started_at
        probability_rows = probabilities.cpu().tolist()
        label_masses = label_mass.cpu().tolist()
        total_input_tokens = int(sum(input_lengths))
        input_tokens_per_second = (
            total_input_tokens / elapsed_seconds
            if elapsed_seconds > 0
            else None
        )
        print(
            json.dumps(
                {
                    "event": "qwen_geval_batch_completed",
                    "batch_size": len(requests),
                    "input_tokens": total_input_tokens,
                    "longest_input_tokens": int(longest_input),
                    "seconds": round(elapsed_seconds, 3),
                    "input_tokens_per_second": (
                        round(input_tokens_per_second, 1)
                        if input_tokens_per_second is not None
                        else None
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        results: list[dict[str, Any]] = []

        for input_length, probability_row, label_probability_mass in zip(
            input_lengths,
            probability_rows,
            label_masses,
            strict=True,
        ):
            weighted_score, distribution, greedy_score = self._label_distribution(
                probability_row
            )
            results.append(
                {
                    "score": weighted_score,
                    "probabilities": distribution,
                    "raw_answer": str(greedy_score),
                    "details": {
                        "backend": "transformers-forward",
                        "attention": self.attention,
                        "label_probability_mass": float(label_probability_mass),
                        "batch_size": len(requests),
                        "input_tokens": int(input_length),
                        "batch_input_tokens": total_input_tokens,
                        "batch_seconds": elapsed_seconds,
                        "batch_input_tokens_per_second": input_tokens_per_second,
                        "logits_to_keep": 1,
                        "prefix_cache_used": False,
                        "prefix_tokens": 0,
                        "computed_tokens": int(input_length),
                    },
                }
            )

        return results

    def _score_resilient(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            return self._score_once(requests)
        except (self.torch.cuda.OutOfMemoryError, RuntimeError) as exception:
            is_out_of_memory = (
                isinstance(exception, self.torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exception).lower()
            )

            if not is_out_of_memory:
                raise

            # The snapshot is the largest thing this service holds beyond the
            # weights, so it is released before retrying rather than competing
            # with the retry for the card. The next prompt rebuilds it.
            held_prefix = self._prefix_cache is not None

            if held_prefix:
                # Not rebuilt for the rest of its group: a snapshot that left
                # too little room for one summary will leave too little for
                # the next.
                self._failed_prefix_ids = self._prefix_ids

            self._reset_prefix_state()

            if len(requests) == 1 and not held_prefix:
                raise

        # Retried only once the except block has ended. Inside it the live
        # traceback keeps every frame of the failed pass reachable - its cache
        # copy and partial activations - so a retry there competed with the
        # very memory it was retrying to escape.
        self._release_memory()

        if len(requests) == 1:
            # The card had room for this prompt before the cache existed, so
            # it is retried without one. The score records that it came from
            # the uncached path.
            return self._score_uncached(requests)

        midpoint = len(requests) // 2

        return (
            self._score_resilient(requests[:midpoint])
            + self._score_resilient(requests[midpoint:])
        )

    def _release_memory(self) -> None:
        """Return everything a failed pass held before trying again."""

        gc.collect()
        self.torch.cuda.empty_cache()

    @modal.method()
    def preview_answer(
        self,
        prompt: str,
        max_new_tokens: int = 24,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Show what the judge writes where scoring reads a digit.

        Scoring reads only the five label logits at the first answer position.
        This returns the judge's greedy continuation from that position and its
        most likely first tokens, so a judge whose `label_probability_mass` is
        low can be seen answering in whatever form it prefers instead.
        Diagnostic only; nothing here feeds a score.
        """

        torch = self.torch
        self._reset_prefix_state()
        formatted = self._format_prompt(prompt)
        ids = self.tokenizer(formatted, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            first = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            probabilities = torch.softmax(first.logits[0, -1].float(), dim=-1)
            top = torch.topk(probabilities, top_k)
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.empty_cache()

        return {
            "input_tokens": len(ids),
            "prompt_tail": formatted[-160:],
            "greedy_answer": self.tokenizer.decode(
                generated[0, len(ids):],
                skip_special_tokens=False,
            ),
            "top_first_tokens": [
                {
                    "token": self.tokenizer.decode([int(token_id)]),
                    "token_id": int(token_id),
                    "probability": round(float(probability), 6),
                }
                for probability, token_id in zip(
                    top.values.tolist(),
                    top.indices.tolist(),
                )
            ],
            "label_probability_mass": float(
                probabilities.index_select(0, self.score_token_ids).sum()
            ),
        }

    @modal.method()
    def ready(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "metric": "geval",
            "revision": QWEN_SCORE_REVISION,
            "judge": JUDGE_NAME,
            "model": JUDGE_MODEL_NAME,
            "transformers": self.transformers_version,
            "attention": self.attention,
            "batch_size": QWEN_SCORE_BATCH_SIZE,
            "max_input_tokens": QWEN_SCORE_MAX_INPUT_TOKENS,
            "prefix_cache": QWEN_SCORE_PREFIX_CACHE,
            "prefix_cache_min_tokens": QWEN_SCORE_PREFIX_CACHE_MIN_TOKENS,
            "prefix_cache_max_tokens": QWEN_SCORE_PREFIX_CACHE_MAX_TOKENS,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []

        if len(requests) > QWEN_SCORE_BATCH_SIZE:
            return [
                {
                    "error": (
                        f"Qwen G-Eval accepts at most "
                        f"{QWEN_SCORE_BATCH_SIZE} prompts per batch."
                    )
                }
                for _ in requests
            ]

        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()

        try:
            return self._score_resilient(requests)
        except Exception as exception:  # noqa: BLE001
            return [
                {
                    "error": f"{type(exception).__name__}: {exception}",
                }
                for _ in requests
            ]
        finally:
            # Memory held before and after each request, and its peak. A
            # `before` that climbs from request to request is memory the
            # service keeps; a high `peak` with a flat `before` is one pass.
            gib = 1024**3
            print(
                json.dumps(
                    {
                        "event": "geval_memory",
                        "allocated_before_gib": round(allocated_before / gib, 2),
                        "allocated_after_gib": round(
                            torch.cuda.memory_allocated() / gib, 2
                        ),
                        "peak_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
                        "prefix_tokens_held": len(self._prefix_ids or []),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


modern_metric_base = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "nltk==3.9.2",
        "numpy==1.26.4",
        "sentencepiece==0.2.0",
        "torch==2.7.1",
        "transformers==4.48.3",
    )
    .run_commands("python -m nltk.downloader punkt punkt_tab")
    .env(REMOTE_CONFIG_ENV)
)

summac_image = (
    modern_metric_base.uv_pip_install(
        "summac==0.0.4",
        # SummaC's package metadata pins huggingface-hub<=0.17.0, which is
        # incompatible with the explicitly pinned Transformers runtime above.
        # SummaC itself works with that newer runtime, so retain our compatible
        # dependencies and install only the SummaC package here.
        extra_options="--no-deps",
    )
    .env(
        {
            "SUMMAC_MODEL_PATH": str(SUMMAC_MODEL_PATH),
            "HF_HUB_OFFLINE": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)

bartscore_image = modern_metric_base.env(
    {
        "BARTSCORE_MODEL_PATH": str(BARTSCORE_MODEL_PATH),
        "HF_HUB_OFFLINE": "1",
    }
).add_local_file(
    SCORING_LOCAL_PATH,
    SCORING_REMOTE_PATH,
    copy=True,
)

supert_image = (
    modern_metric_base.apt_install("git")
    .uv_pip_install(
        "networkx==3.4.2",
        "scikit-learn==1.6.1",
        f"sentence-transformers=={SUPERT_SENTENCE_TRANSFORMERS_VERSION}",
    )
    .run_commands(
        (
            "git clone https://github.com/yg211/acl20-ref-free-eval "
            "/opt/acl20-ref-free-eval"
        ),
        (
            "git -C /opt/acl20-ref-free-eval checkout "
            "f283ce2c80ef8fe81f83d4c09c68b9580eaee570"
        ),
        "python -m nltk.downloader punkt stopwords",
        (
            "python -c \"import sys; import sentence_transformers as st; "
            f"assert st.__version__ == '{SUPERT_SENTENCE_TRANSFORMERS_VERSION}'; "
            "sys.path.insert(0, '/opt/acl20-ref-free-eval'); "
            "import ref_free_metrics.supert as supert; "
            "assert supert.SentenceTransformer is st.SentenceTransformer\""
        ),
    )
    .env(
        {
            "SUPERT_HOME": "/opt/acl20-ref-free-eval",
            "SUPERT_MODEL_PATH": str(SUPERT_MODEL_PATH),
            "HF_HUB_OFFLINE": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)

# Installed with --no-deps, like SummaC: AlignScore pins torch<2 and numpy<2,
# which would replace the base's runtime, and runs on it unchanged. Its own
# requirements are added explicitly. pytorch-lightning stays below 2, where
# `load_from_checkpoint` may still be called on an instance as AlignScore does;
# that release imports pkg_resources, hence setuptools.
alignscore_image = (
    modern_metric_base.apt_install("git")
    .uv_pip_install(
        "pytorch-lightning==1.9.5",
        "scikit-learn==1.6.1",
        "setuptools==75.8.0",
        "spacy==3.7.5",
        (
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/"
            "download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
        ),
    )
    .uv_pip_install(
        f"alignscore @ git+https://github.com/yuh-zha/AlignScore@{ALIGNSCORE_COMMIT}",
        extra_options="--no-deps",
    )
    .env(
        {
            "ALIGNSCORE_CHECKPOINT_PATH": str(ALIGNSCORE_CHECKPOINT_PATH),
            "HF_HUB_CACHE": str(HUB_MODEL_CACHE),
            "HF_HUB_OFFLINE": "1",
            # A Lightning 1.x checkpoint pickles its hyperparameters, which
            # torch>=2.6's weights-only load refuses. The file is pinned above.
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)

# Without its `llm` extra: vLLM is imported only for the 7B checker.
minicheck_image = (
    modern_metric_base.apt_install("git")
    .uv_pip_install("accelerate==1.3.0")
    .uv_pip_install(
        f"minicheck @ git+https://github.com/Liyan06/MiniCheck@{MINICHECK_COMMIT}",
        extra_options="--no-deps",
    )
    .env(
        {
            "HF_HUB_CACHE": str(HUB_MODEL_CACHE),
            "HF_HUB_OFFLINE": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)

# Bespoke-MiniCheck-7B builds on the generation image for its CUDA torch.
# The library would serve it through vLLM, which misloads this architecture,
# so the backend runs the model's own reference code instead (see
# BespokeMiniCheckBackend). transformers is pinned to the last 4.x, which that
# reference code and its tokenizer are written against.
minicheck_7b_image = (
    qwen_image.apt_install("git")
    .uv_pip_install("nltk==3.9.2", "transformers==4.57.6")
    .run_commands("python -m nltk.downloader punkt punkt_tab")
    .uv_pip_install(
        f"minicheck @ git+https://github.com/Liyan06/MiniCheck@{MINICHECK_COMMIT}",
        extra_options="--no-deps",
    )
    .env(
        {
            "HF_HUB_CACHE": str(HUB_MODEL_CACHE),
            "HF_HUB_OFFLINE": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)

# The image's own interpreter, which owns the QAFactEval stack. `add_python`
# below puts a newer Python on PATH for Modal's own use, so every install here
# names this one explicitly rather than whichever `pip` resolves first.
LEGACY_METRIC_PYTHON = "/opt/conda/bin/python"

qafacteval_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime",
        # This image ships Python 3.7 and the Modal client injected into every
        # container refuses to import below 3.10, so the container used to die
        # at startup with "This version of Modal requires at least Python
        # 3.10". The metric's stack cannot move to meet it - qaeval 0.1.0 pins
        # torch==1.6.0, spacy==2.2.4, allennlp==1.1.0 and transformers==3.0.2,
        # none of which have wheels above 3.8 - so the image carries both
        # interpreters instead: Modal runs on this one, the metric stays on
        # LEGACY_METRIC_PYTHON, and QAFactEvalService bridges them.
        add_python="3.11",
        # `add_python` only unpacks the interpreter into /usr/local and links
        # /usr/local/bin/python; it leaves PATH alone. This base image puts
        # /opt/conda/bin first, so `python` still resolved to its 3.7 and the
        # runner crash-looped on the very error add_python was meant to fix.
        # These commands are injected directly after add_python's own, which is
        # the only point where the ordering can still be corrected.
        setup_dockerfile_commands=[
            "ENV PATH=/usr/local/bin:$PATH",
            # Fail the build rather than ship an image that would crash-loop.
            'RUN python -c "import sys; assert sys.version_info[:2] >= (3, 10),'
            ' sys.version" && test -x /opt/conda/bin/python',
        ],
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "git",
    )
    .run_commands(
        ("git clone https://github.com/salesforce/QAFactEval /opt/QAFactEval"),
        ("git -C /opt/QAFactEval checkout 01177f11cc05f8b3e75511a0cfa54bac19324d09"),
        (
            f"{LEGACY_METRIC_PYTHON} -m pip install --only-binary=:all: "
            "urllib3==1.25.11 "
            "murmurhash==1.0.2 "
            "cymem==2.0.4 "
            "preshed==3.0.2 "
            "blis==0.4.1 "
            "srsly==1.0.2"
        ),
        f"{LEGACY_METRIC_PYTHON} -m pip install jsonnet==0.16.0",
        f"{LEGACY_METRIC_PYTHON} -m pip install qaeval==0.1.0",
        f"{LEGACY_METRIC_PYTHON} -m pip install --no-deps -e /opt/QAFactEval",
    )
    .env(
        {
            **REMOTE_CONFIG_ENV,
            "QAFACTEVAL_HOME": "/opt/QAFactEval",
            "QAFACTEVAL_MODEL_DIR": str(QAFACTEVAL_MODEL_PATH),
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)


# FactGraph's stack, which is its own and not this project's: Python 3.7 and
# torch 1.9 on CUDA 11.1 with PyTorch Geometric, as the authors' README
# installs it, and the repository's patched copy of transformers 4.6.1, which
# its pickled checkpoint needs to unpickle. The base image's interpreter runs
# it, behind LegacyMetricProcess, exactly as for QAFactEval.
FACTGRAPH_HOME = "/opt/fact-graph"
FACTGRAPH_PREPROCESS = f"{FACTGRAPH_HOME}/data/preprocess"
# Where the authors' align_amr looks for the fast_align binaries.
FACTGRAPH_FAST_ALIGN = f"{FACTGRAPH_PREPROCESS}/fast_align"

factgraph_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime",
        # Python 3.7 again, so the same two-interpreter arrangement as
        # qafacteval_image; see the comments there.
        add_python="3.11",
        setup_dockerfile_commands=[
            "ENV PATH=/usr/local/bin:$PATH",
            'RUN python -c "import sys; assert sys.version_info[:2] >= (3, 10),'
            ' sys.version" && test -x /opt/conda/bin/python',
        ],
    )
    .entrypoint([])
    .apt_install("build-essential", "cmake", "git")
    .run_commands(
        f"git clone https://github.com/amazon-science/fact-graph {FACTGRAPH_HOME}",
        f"git -C {FACTGRAPH_HOME} checkout {FACTGRAPH_COMMIT}",
        f"git clone https://github.com/clab/fast_align {FACTGRAPH_FAST_ALIGN}",
        f"git -C {FACTGRAPH_FAST_ALIGN} checkout {FAST_ALIGN_COMMIT}",
        # The base is Ubuntu 18.04, whose cmake (3.10) predates `-S`/`-B`, so
        # this is the build FactGraph's own setup script runs.
        f"mkdir -p {FACTGRAPH_FAST_ALIGN}/build && cd {FACTGRAPH_FAST_ALIGN}/build "
        "&& cmake .. && make",
        "git clone https://github.com/ablodge/amr-utils /opt/amr-utils",
        f"git -C /opt/amr-utils checkout {AMR_UTILS_COMMIT}",
        (
            f"{LEGACY_METRIC_PYTHON} -m pip install torch-scatter==2.0.9 torch-sparse==0.6.12 "
            "-f https://data.pyg.org/whl/torch-1.9.0+cu111.html"
        ),
        (
            # transformers and tokenizers at the versions the repository's own
            # copy declares, so the installed metadata agrees with the copy
            # that shadows it at run time.
            f"{LEGACY_METRIC_PYTHON} -m pip install "
            "torch-geometric==2.0.3 rdflib==6.1.1 "
            "transformers==4.6.1 tokenizers==0.10.3 "
            "sentence-transformers==1.2.0 amrlib==0.7.1 "
            "spacy==3.0.6 penman==1.2.0 smatch==1.0.4 unidecode==1.3.3 "
            "scikit-learn==1.0.2 scipy==1.7.3 "
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl"
        ),
        f"{LEGACY_METRIC_PYTHON} -m pip install --no-deps /opt/amr-utils",
        # Fail the build, not the first scoring call: torch must still be the
        # base image's, amrlib's aligner must find fast_align (this also unpacks
        # the aligner model amrlib ships), and the modules must import in the
        # order FactGraphBackend uses, with `transformers` resolving to the
        # repository's copy.
        f"{LEGACY_METRIC_PYTHON} -c \"import torch; assert torch.__version__.startswith('1.9.0'), torch.__version__\"",
        (
            f"{LEGACY_METRIC_PYTHON} -c \"from amrlib.alignments.faa_aligner import FAA_Aligner; "
            f"FAA_Aligner(bin_dir='{FACTGRAPH_FAST_ALIGN}/build')\""
        ),
        (
            f"cd /tmp && {LEGACY_METRIC_PYTHON} -c \"import sys; "
            f"sys.path[:0] = ['{FACTGRAPH_PREPROCESS}', '{FACTGRAPH_HOME}/src']; "
            "import amrlib, augmentation_ops, preprocess_evaluate; "
            "sys.modules.pop('utils'); "
            f"sys.path.remove('{FACTGRAPH_PREPROCESS}'); "
            "import models, utils_evaluate, transformers, torch_geometric; "
            f"assert transformers.__file__.startswith('{FACTGRAPH_HOME}/src'), transformers.__file__\""
        ),
    )
    .env(
        {
            **REMOTE_CONFIG_ENV,
            "FACTGRAPH_HOME": FACTGRAPH_HOME,
            "FACTGRAPH_MODEL_DIR": str(FACTGRAPH_MODEL_PATH),
            # amrlib's parser reads the tokenizer of the checkpoint it was
            # trained from by Hub name; QAFactEval's pinned copy of that same
            # checkpoint stands in for it offline.
            "FACTGRAPH_BART_TOKENIZER": str(QAFACTEVAL_BASE_MODEL_PATH),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_file(
        SCORING_LOCAL_PATH,
        SCORING_REMOTE_PATH,
        copy=True,
    )
)


def metric_options(modal_class: str) -> dict[str, Any]:
    """Constructor-time options for the metrics one Modal class serves.

    `parameters` in experiment.yaml travel with each scoring request; `options`
    configure the model itself and so are only readable at load time, when the
    container has the config but not yet any request. Two metrics may share one
    class (BARTScore's two directions), so their options are merged and a
    genuine disagreement is an error rather than a silent last-wins.
    """

    metrics = EXPERIMENT_CONFIG.get("metrics")

    if metrics is None:
        # Without this the backends would fall back to their own defaults and
        # score on a configuration nobody chose, which is exactly the silent
        # failure the revision check exists to catch late.
        raise RuntimeError(
            "This container received no `metrics` block in "
            f"{CONFIG_ENV_NAME}, so no metric could be configured from "
            "experiment.yaml. Redeploy with a modal_app.py that serializes it."
        )

    options: dict[str, Any] = {}

    for name, specification in metrics.items():
        if specification.get("modal_class") != modal_class:
            continue

        for key, value in (specification.get("options") or {}).items():
            if key in options and options[key] != value:
                raise ValueError(
                    f"{modal_class} serves metrics that disagree on option "
                    f"{key!r}: {options[key]!r} against {name}'s {value!r}."
                )

            options[key] = value

    return options


def load_metric_backend(metric: str, modal_class: str) -> Any:
    """Load only the backend half of scoring.py inside a metric image."""

    os.chdir(MODEL_ROOT)
    specification = importlib.util.spec_from_file_location(
        "deception_metric_backends",
        SCORING_REMOTE_PATH,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the metric backend module.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module.build_external_backend(
        metric=metric,
        device="cuda:0",
        options=metric_options(modal_class),
    )


class LegacyMetricProcess:
    """A metric run beside this container's interpreter rather than inside it.

    `scoring.py` already speaks a persistent JSONL protocol on stdin/stdout for
    metrics that must stay isolated, so the bridge is that protocol pointed at
    the image's own Python instead of a separate machine. The metric keeps its
    pinned 2020 stack and its own batching; Modal keeps a Python it can import.
    """

    # Variables that tell an interpreter where its own standard library and
    # packages live. This container's Modal runs on a different Python, so
    # inheriting them points the worker at the wrong installation and it dies
    # before reaching any project code.
    INTERPRETER_ENVIRONMENT = (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONNOUSERSITE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    )

    def __init__(
        self,
        metric: str,
        python: str,
        cwd: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.metric = metric
        self.python = python
        self.cwd = cwd
        self.options = options or {}
        self.process: Any = None
        self.revision: str | None = None
        self.diagnostics: list[str] = []
        self.reader: Any = None

    def _environment(self) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in self.INTERPRETER_ENVIRONMENT
        }
        # The worker imports torch from the image's own installation, so it must
        # not inherit a site-packages path belonging to another interpreter.
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"

        return environment

    def _drain(self, stream: Any) -> None:
        """Mirror the worker's stderr into this container's log, and keep it.

        Keeping a bounded tail is what turns "the container crash-looped" into
        an error message that names the cause, since a failed `@modal.enter()`
        gives the caller no other view of the subprocess.
        """

        for line in stream:
            sys.stderr.write(f"[{self.metric}] {line}")
            self.diagnostics.append(line.rstrip("\n"))
            del self.diagnostics[:-40]

    # Enough to get past any banner a 2020 dependency prints, and low enough
    # that a worker babbling on stdout fails instead of hanging the container.
    MAXIMUM_PREAMBLE_LINES = 500

    def _read_message(self, expectation: str) -> dict[str, Any]:
        """Read the next JSON line, stepping over non-JSON output.

        `external_metric_main` redirects the backend's `sys.stdout` to stderr,
        but that only rebinds Python's own stream: a C extension writing to
        file descriptor 1 goes straight into this pipe regardless, and the
        pinned 2020 stack has several. Such a line is logged and skipped rather
        than parsed, and - unlike a bare `json.loads` - it is never discarded,
        so a worker that fails here says what it actually printed.
        """

        preamble: list[str] = []

        while True:
            line = self.process.stdout.readline()

            if not line:
                raise self._failure(
                    f"The {self.metric} worker closed its output before "
                    f"sending {expectation}."
                )

            stripped = line.strip()

            if not stripped:
                continue

            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                sys.stderr.write(f"[{self.metric}] stdout: {line}")
                preamble.append(stripped)
                self.diagnostics.append(f"stdout: {stripped}")
                del self.diagnostics[:-40]

                if len(preamble) >= self.MAXIMUM_PREAMBLE_LINES:
                    raise self._failure(
                        f"The {self.metric} worker wrote "
                        f"{len(preamble)} lines of non-JSON output instead of "
                        f"{expectation}."
                    ) from None

                continue

            if not isinstance(message, dict):
                raise self._failure(
                    f"The {self.metric} worker sent {type(message).__name__} "
                    f"where {expectation} was expected: {stripped[:200]}"
                )

            return message

    def _failure(self, summary: str) -> RuntimeError:
        # Wait first: the exit code is the single most useful fact here and it
        # is not set until the process is reaped. Then let the reader thread
        # finish, so a traceback written just before the exit is not lost to a
        # race with this message.
        code = None

        if self.process is not None:
            code = self.process.wait()

        if self.reader is not None:
            self.reader.join(timeout=5)

        tail = "\n".join(self.diagnostics[-25:])

        return RuntimeError(
            f"{summary}"
            + (f" Exit code: {code}." if code is not None else "")
            + f"\nInterpreter: {self.python}\n"
            + (f"Last output from the worker:\n{tail}" if tail else
               "The worker produced no output.")
        )

    def start(self) -> None:
        if self.process is not None:
            return

        if not Path(self.python).is_file():
            raise RuntimeError(
                f"No interpreter at {self.python}; the {self.metric} image is "
                "expected to keep its original Python alongside the one Modal "
                "runs on."
            )

        self.process = subprocess.Popen(
            [
                self.python,
                SCORING_REMOTE_PATH,
                "--metric",
                self.metric,
                "--options",
                json.dumps(self.options),
            ],
            cwd=self.cwd,
            env=self._environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(
            target=self._drain,
            args=(self.process.stderr,),
            daemon=True,
        )
        self.reader.start()

        announcement = self._read_message("its startup handshake")

        if announcement.get("status") != "ready":
            raise self._failure(
                announcement.get("error", f"{self.metric} worker failed to start.")
            )

        self.revision = announcement["revision"]

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.start()

        if not requests:
            return []

        self.process.stdin.write(json.dumps({"batch": requests}) + "\n")
        self.process.stdin.flush()
        response = self._read_message("a batch result")

        if "error" in response:
            raise RuntimeError(response["error"])

        results = response["results"]

        if len(results) != len(requests):
            raise RuntimeError(
                f"{self.metric} returned {len(results)} results for "
                f"{len(requests)} requests."
            )

        return results


def score_metric_batch(
    backend: Any,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score a source-local GPU batch and return structured failures."""

    if not requests:
        return []

    parameters = requests[0].get("parameters", {})

    if any(request.get("parameters", {}) != parameters for request in requests):
        return [
            {
                "error": (
                    "ValueError: one Modal metric batch contained "
                    "different parameter sets."
                )
            }
            for _ in requests
        ]

    pairs = [
        (
            request["source_text"],
            request["summary_text"],
        )
        for request in requests
    ]

    try:
        return backend.score_many(
            pairs=pairs,
            parameters=parameters,
        )
    except Exception as exception:  # noqa: BLE001
        return [
            {
                "error": f"{type(exception).__name__}: {exception}",
            }
            for _ in requests
        ]


@app.cls(
    image=summac_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class SummaCService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("summac", "SummaCService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "summac",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


# SummaC-ZS shares SummaCService's image and NLI checkpoint but needs its own
# class: `metric_options` merges the options of every metric a class serves and
# refuses a disagreement, and these two disagree on `granularity` by design -
# ZS runs at the paper's sentence granularity, which Conv's trained head cannot
# take on sources this long. See SummaCZeroShotBackend.
@app.cls(
    image=summac_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class SummaCZeroShotService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("summac_zs", "SummaCZeroShotService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "summac_zs",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=bartscore_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class BARTScoreService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("bartscore", "BARTScoreService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "bartscore",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=supert_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class SupertService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("supert", "SupertService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "supert",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=qafacteval_image,
    gpu=LEGACY_METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class QAFactEvalService:
    @modal.enter()
    def load(self) -> None:
        # Unlike the other three, this backend is not imported here: it cannot
        # run on the Python this container runs on. See LegacyMetricProcess,
        # which forwards these options to the worker's own `--options`.
        self.worker = LegacyMetricProcess(
            metric="qafacteval",
            python=LEGACY_METRIC_PYTHON,
            cwd=str(MODEL_ROOT),
            options=metric_options("QAFactEvalService"),
        )
        self.worker.start()

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "qafacteval",
            "revision": self.worker.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []

        parameters = requests[0].get("parameters", {})

        if any(request.get("parameters", {}) != parameters for request in requests):
            return [
                {
                    "error": (
                        "ValueError: one Modal metric batch contained "
                        "different parameter sets."
                    )
                }
                for _ in requests
            ]

        try:
            return self.worker.score_batch(requests)
        except Exception as exception:  # noqa: BLE001
            return [
                {"error": f"{type(exception).__name__}: {exception}"}
                for _ in requests
            ]


@app.cls(
    image=factgraph_image,
    gpu=FACTGRAPH_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class FactGraphService:
    @modal.enter()
    def load(self) -> None:
        # Runs on the image's Python 3.7, as QAFactEvalService does.
        self.worker = LegacyMetricProcess(
            metric="factgraph",
            python=LEGACY_METRIC_PYTHON,
            cwd=str(MODEL_ROOT),
            options=metric_options("FactGraphService"),
        )
        self.worker.start()

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "factgraph",
            "revision": self.worker.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []

        try:
            return self.worker.score_batch(requests)
        except Exception as exception:  # noqa: BLE001
            return [
                {"error": f"{type(exception).__name__}: {exception}"}
                for _ in requests
            ]


@app.cls(
    image=alignscore_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class AlignScoreService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("alignscore", "AlignScoreService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "alignscore",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=minicheck_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class MiniCheckService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("minicheck", "MiniCheckService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "minicheck",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


# The authors' own chunking, as an ablation of the review-packed setting above.
# A separate class because `metric_options` merges the options of every metric
# one class serves, and these two disagree on `chunking` by design - the same
# reason SummaC-ZS has its own class.
@app.cls(
    image=alignscore_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class AlignScoreDefaultService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("alignscore", "AlignScoreDefaultService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "alignscore",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=minicheck_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class MiniCheckDefaultService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("minicheck", "MiniCheckDefaultService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "minicheck",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


# The authors' SummaC settings - sentence premises, the first 100 kept - as
# ablations of the one-review-per-premise setting. Own classes for the reason
# AlignScoreDefaultService gives.
@app.cls(
    image=summac_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class SummaCDefaultService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("summac", "SummaCDefaultService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "summac",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=summac_image,
    gpu=METRIC_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class SummaCZeroShotDefaultService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("summac_zs", "SummaCZeroShotDefaultService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "summac_zs",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


# A 7B model with a 32k context: the generation GPU rather than the small
# metric one.
@app.cls(
    image=minicheck_7b_image,
    gpu=QWEN_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class MiniCheck7BService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("minicheck_7b", "MiniCheck7BService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "minicheck_7b",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)


@app.cls(
    image=minicheck_7b_image,
    gpu=QWEN_GPU,
    timeout=60 * MINUTE,
    startup_timeout=30 * MINUTE,
    scaledown_window=METRIC_SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={
        str(MODEL_ROOT): read_only_model_weights,
    },
)
class MiniCheck7BDefaultService:
    @modal.enter()
    def load(self) -> None:
        self.backend = load_metric_backend("minicheck_7b", "MiniCheck7BDefaultService")

    @modal.method()
    def ready(self) -> dict[str, str]:
        return {
            "status": "ready",
            "metric": "minicheck_7b",
            "revision": self.backend.revision,
        }

    @modal.method()
    def score_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_metric_batch(self.backend, requests)
