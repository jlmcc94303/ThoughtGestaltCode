import argparse
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from transformers import GPT2Tokenizer
from src_recurrent.core.tokenizer_setup import build_tokenizer

from src_recurrent.core.data.sentence_splitter import create_token_based_sentence_splitter
from src_recurrent.pipelines.data import (
    _is_top_title,
    extract_all_documents_from_files,
    process_documents_for_training,
    chunk_processed_documents,
    compute_full_dataset_analysis,
    compute_essential_dataset_stats,
    write_analysis_files,
)


logger = logging.getLogger("subset_dataset")

TEXT_EXTENSIONS = {
    ".txt",
    ".train",
    ".md",
    ".text",
    ".tokens",
    ".tokens.txt",
}


@dataclass
class FileDocuments:
    path: str
    relative_path: str
    kind: str  # "text" or "jsonl"
    dataset: str
    documents: List[Dict[str, Any]]
    total_tokens: int
    preserve_full_file: bool = False


def _read_all_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="replace") as fh:
            return fh.readlines()


def _read_text_documents(path: str, tokenizer) -> List[Dict[str, Any]]:
    lines = _read_all_lines(path)
    documents: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    have_title = False

    def _flush() -> None:
        nonlocal current_lines, have_title
        if not current_lines:
            return
        text = "".join(current_lines).strip()
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        documents.append({
            "lines": list(current_lines),
            "token_count": token_count,
            "text": text,
        })
        current_lines = []
        have_title = False

    for raw in lines:
        stripped = raw.strip()
        if _is_top_title(stripped):
            _flush()
            current_lines = [raw]
            have_title = True
        else:
            if have_title:
                current_lines.append(raw)

    _flush()
    return documents


def _read_preseparated_document(path: str, tokenizer) -> List[Dict[str, Any]]:
    lines = _read_all_lines(path)
    text = "".join(lines)
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    return [{
        "lines": list(lines),
        "token_count": token_count,
        "text": text,
    }]


def _read_jsonl_documents(path: str, tokenizer) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_idx, raw in enumerate(fh):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line %s in %s", line_idx + 1, path)
                continue
            if isinstance(payload, dict):
                if "text" in payload:
                    text = payload["text"]
                elif "question" in payload and "answer" in payload:
                    text = f"Question: {payload['question']} Answer: {payload['answer']}"
                else:
                    text = " ".join(v for v in payload.values() if isinstance(v, str) and v.strip())
            else:
                text = str(payload)
            text = (text or "").strip()
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            documents.append({
                "json": stripped,
                "token_count": token_count,
                "text": text,
            })
    return documents


def _gather_split_documents(
    split_root: str,
    tokenizer,
    rng: Optional[random.Random] = None,
    preseparated_docs: bool = False,
    dataset_label: Optional[str] = None,
) -> Tuple[List[FileDocuments], int]:
    if not os.path.isdir(split_root):
        return [], 0

    gathered: List[FileDocuments] = []
    total_tokens = 0

    for dirpath, _, filenames in os.walk(split_root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(path, split_root)
            dataset_name = rel_path.split(os.sep)[0]
            ext = os.path.splitext(name)[1].lower()
            if ext == ".jsonl":
                docs = _read_jsonl_documents(path, tokenizer)
                kind = "jsonl"
            elif ext in TEXT_EXTENSIONS or ext.endswith(".txt"):
                if preseparated_docs:
                    docs = _read_preseparated_document(path, tokenizer)
                else:
                    docs = _read_text_documents(path, tokenizer)
                kind = "text"
            else:
                logger.info("Skipping unsupported file %s", path)
                continue
            if rng is not None and docs:
                rng.shuffle(docs)
            file_total = sum(doc.get("token_count", 0) for doc in docs)
            total_tokens += file_total
            gathered.append(FileDocuments(
                path=path,
                relative_path=rel_path,
                kind=kind,
                dataset=dataset_label or dataset_name,
                documents=docs,
                total_tokens=file_total,
                preserve_full_file=preseparated_docs,
            ))
            logger.info("Loaded %s (%s docs, %s tokens)", rel_path, len(docs), file_total)
    if preseparated_docs:
        if rng is not None:
            rng.shuffle(gathered)
        else:
            random.shuffle(gathered)
    else:
        gathered.sort(key=lambda x: x.relative_path)
    return gathered, total_tokens


def _compute_dataset_totals(split_docs: List[FileDocuments]) -> Dict[str, int]:
    totals: Dict[str, int] = defaultdict(int)
    for fd in split_docs:
        totals[fd.dataset] += fd.total_tokens
    return dict(totals)




def _select_documents(documents: List[Dict[str, Any]], target_tokens: float) -> Tuple[List[Dict[str, Any]], int]:
    if target_tokens <= 0:
        return [], 0
    selected: List[Dict[str, Any]] = []
    total = 0
    idx = 0
    while idx < len(documents):
        if total >= target_tokens:
            break
        doc = documents[idx]
        selected.append(doc)
        total += doc.get("token_count", 0)
        if total >= target_tokens:
            overshoot = total - target_tokens
            without = total - doc.get("token_count", 0)
            if selected and abs(without - target_tokens) < abs(total - target_tokens):
                selected.pop()
                total = without
                idx += 1
                continue
            break
        idx += 1
    return selected, total


def _write_subset_file(out_path: str, kind: str, documents: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if kind == "jsonl":
        with open(out_path, "w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(doc["json"] + "\n")
    elif kind == "text":
        with open(out_path, "w", encoding="utf-8") as fh:
            for doc in documents:
                fh.writelines(doc["lines"])
    else:
        raise ValueError(f"Unsupported file kind: {kind}")


def _format_tokens(n: float) -> str:
    return f"{int(n):,}"


def _generate_subsets(
    split_docs: List[FileDocuments],
    split_name: str,
    scale: float,
    output_root: str,
    summary: Dict[str, Any],
) -> Tuple[int, Dict[str, int]]:
    actual_total = 0
    dataset_scale = max(0.0, min(scale, 1.0))
    split_summary = {
        "scale": dataset_scale,
        "files": [],
        "actual_tokens": 0,
        "requested_tokens": 0,
        "original_tokens": sum(fd.total_tokens for fd in split_docs),
    }

    dataset_original_totals = _compute_dataset_totals(split_docs)
    dataset_target_totals = {
        name: int(round(total * dataset_scale))
        for name, total in dataset_original_totals.items()
    }
    dataset_requested_totals: Dict[str, int] = defaultdict(int)
    dataset_actual_totals: Dict[str, int] = defaultdict(int)
    for fd in split_docs:
        target_tokens = fd.total_tokens * dataset_scale
        requested = int(round(target_tokens))
        dataset_requested_totals[fd.dataset] += requested

        if dataset_scale >= 0.9999:
            selected_docs = fd.documents
            actual_tokens = fd.total_tokens
            requested = fd.total_tokens
        elif fd.preserve_full_file:
            dataset_target = dataset_target_totals.get(fd.dataset, 0)
            remaining = dataset_target - dataset_actual_totals[fd.dataset]
            if remaining <= 0:
                selected_docs = []
                actual_tokens = 0
            elif remaining >= fd.total_tokens:
                selected_docs = fd.documents
                actual_tokens = fd.total_tokens
            else:
                include = abs(remaining - fd.total_tokens) < abs(remaining)
                if include:
                    selected_docs = fd.documents
                    actual_tokens = fd.total_tokens
                else:
                    selected_docs = []
                    actual_tokens = 0
        else:
            selected_docs, actual_tokens = _select_documents(fd.documents, target_tokens)
            if dataset_scale >= 0.9999 or actual_tokens >= fd.total_tokens:
                selected_docs = fd.documents
                actual_tokens = fd.total_tokens
                requested = fd.total_tokens

        actual_total += actual_tokens
        dataset_actual_totals[fd.dataset] += actual_tokens

        out_path = os.path.join(output_root, split_name, fd.relative_path)
        if selected_docs:
            _write_subset_file(out_path, fd.kind, selected_docs)
        elif os.path.exists(out_path):
            os.remove(out_path)

        split_summary["files"].append({
            "relative_path": fd.relative_path,
            "original_tokens": fd.total_tokens,
            "requested_tokens": requested,
            "actual_tokens": actual_tokens,
            "documents_selected": len(selected_docs),
            "documents_total": len(fd.documents),
            "dataset": fd.dataset,
            "dataset_scale": dataset_scale,
            "preserve_full_file": fd.preserve_full_file,
        })
        split_summary["requested_tokens"] += requested

    split_summary["actual_tokens"] = actual_total
    split_summary["datasets"] = []
    dataset_keys = sorted(dataset_original_totals.keys())
    for dataset in dataset_keys:
        original_tokens = dataset_original_totals[dataset]
        requested_tokens = dataset_requested_totals.get(dataset, 0)
        actual_tokens = dataset_actual_totals.get(dataset, 0)
        split_summary["datasets"].append({
            "name": dataset,
            "original_tokens": original_tokens,
            "requested_tokens": requested_tokens,
            "actual_tokens": actual_tokens,
            "scale": dataset_scale,
        })

    summary[split_name] = split_summary
    return actual_total, dict(dataset_actual_totals)


def _run_analysis(
    dataset_root: str,
    split_name: str,
    tokenizer: GPT2Tokenizer,
    splitter,
    analysis_dir: str,
    chunk_size: int,
    min_sentences: int,
    max_sentence_tokens: int,
    min_sentence_tokens: int,
    use_chunking: bool,
    cache_key: str,
    sentence_tail_len: int,
) -> None:
    split_path = os.path.join(dataset_root, split_name)
    if not os.path.isdir(split_path):
        logger.info("Skipping analysis for %s (missing directory)", split_name)
        return
    gathered_files = []
    for root, _, files in os.walk(split_path):
        for name in files:
            lower = name.lower()
            if lower.endswith((".txt", ".jsonl", ".train", ".md", ".tokens", ".tokens.txt")):
                gathered_files.append(os.path.join(root, name))
    if not gathered_files:
        logger.info("No files found for analysis in %s", split_path)
        return
    gathered_files.sort()
    logger.info("Running analysis for %s (%d files)", split_name, len(gathered_files))

    docs = extract_all_documents_from_files(gathered_files, splitter, use_document_boundaries=True)
    if not docs:
        logger.info("No documents extracted for %s", split_name)
        return

    processed = process_documents_for_training(
        docs,
        tokenizer=tokenizer,
        sentence_splitter=splitter,
        max_sentence_tokens=max_sentence_tokens,
        min_sentences_per_document=min_sentences,
        min_sentence_tokens_filter=min_sentence_tokens,
        sentence_tail_len=sentence_tail_len,
    )

    if use_chunking:
        examples = chunk_processed_documents(
            processed,
            chunk_size=chunk_size,
            split_name=split_name,
        )
    else:
        examples = processed

    analysis_full = compute_full_dataset_analysis(split_name, examples)
    write_analysis_files(analysis_dir, split_name, cache_key, analysis_full)

    analysis_ess = compute_essential_dataset_stats(split_name, processed, examples)
    write_analysis_files(analysis_dir, f"{split_name}.essentials", cache_key, analysis_ess)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create composition-preserving dataset subsets by token count.")
    parser.add_argument("data_folder", type=str, help="Path to the original data folder containing train/ and val/ directories.")
    parser.add_argument("target_train_millions", type=float, help="Desired train token count in millions (e.g. 10 for 10M tokens).")
    parser.add_argument(
        "target_val_millions",
        type=float,
        nargs="?",
        default=None,
        help="Optional validation token target in millions. If omitted, the full val split is copied.",
    )
    parser.add_argument("--output-root", type=str, default=None, help="Destination directory for subsets (defaults to alongside data folder).")
    parser.add_argument("--min-sentences-per-document", type=int, default=0)
    parser.add_argument("--max-sentence-tokens", type=int, default=64)
    parser.add_argument("--sentence-tail-len", type=int, default=1, help="Tail tokens per sentence (EOS plus optional reserved slots).")
    parser.add_argument("--min-sentence-tokens", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--no-chunking", type=bool, default=False, help="Disable chunking when producing analysis outputs.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output directory.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed used to shuffle documents before subsetting.")
    parser.add_argument(
        "--preseparated-docs",
        action="store_true",
        help="Treat each source file as a standalone document (e.g., WebText-style corpora).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    data_folder = os.path.abspath(args.data_folder)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    base_name = os.path.basename(os.path.normpath(data_folder))
    if args.target_train_millions >= 1:
        target_suffix = f"{int(round(args.target_train_millions))}M"
    else:
        target_suffix = f"{args.target_train_millions:.2f}M".replace(".", "p")

    default_output_root = os.path.dirname(data_folder)
    output_root = os.path.abspath(args.output_root or os.path.join(default_output_root, f"{base_name}_{target_suffix}"))

    if os.path.exists(output_root):
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite to replace it.")
    else:
        os.makedirs(output_root, exist_ok=True)

    logger.info("Preparing subset in %s", output_root)

    # Initialize tokenizer for token counting
    tokenizer = build_tokenizer()
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "bos_token": "[BOS]",
    })
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "[BOS]"

    rng_train = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else None
    rng_val = random.Random(args.shuffle_seed + 1) if args.shuffle_seed is not None else None

    train_docs, train_total = _gather_split_documents(
        os.path.join(data_folder, "train"),
        tokenizer,
        rng_train,
        preseparated_docs=args.preseparated_docs,
        dataset_label=base_name if args.preseparated_docs else None,
    )
    train_dataset_original = _compute_dataset_totals(train_docs)

    target_train_tokens = int(args.target_train_millions * 1_000_000)
    summary: Dict[str, Any] = {
        "data_folder": data_folder,
        "output_folder": output_root,
        "target_train_tokens": target_train_tokens,
        "train_total_tokens": train_total,
        "train_dataset_original_tokens": train_dataset_original,
        "target_val_tokens": None,
        "val_subset_requested": args.target_val_millions is not None,
        "preseparated_docs": args.preseparated_docs,
    }

    if train_total == 0:
        logger.warning("Train split has zero tokens; nothing to subset.")
        train_scale = 0.0
    else:
        desired_scale = target_train_tokens / train_total
        if desired_scale >= 1.0:
            logger.warning(
                "Requested train token count (%s) exceeds available tokens (%s). Using full dataset.",
                _format_tokens(target_train_tokens),
                _format_tokens(train_total),
            )
            train_scale = 1.0
        else:
            train_scale = desired_scale
    summary["train_scale"] = train_scale

    actual_train_tokens, train_dataset_actual = _generate_subsets(
        train_docs,
        "train",
        train_scale,
        output_root,
        summary,
    )
    summary["actual_train_tokens"] = actual_train_tokens
    summary["train_dataset_actual_tokens"] = train_dataset_actual

    val_source = os.path.join(data_folder, "val")
    val_dest = os.path.join(output_root, "val")
    summary["val_subset"] = False
    if not os.path.isdir(val_source):
        logger.warning("No validation directory found at %s", val_source)
        summary["val_copied"] = False
    else:
        if args.target_val_millions is None:
            logger.info("Copying validation directory as-is from %s to %s", val_source, val_dest)
            if os.path.exists(val_dest):
                shutil.rmtree(val_dest)
            shutil.copytree(val_source, val_dest)
            summary["val_copied"] = True
        else:
            val_docs, val_total = _gather_split_documents(
                val_source,
                tokenizer,
                rng_val,
                preseparated_docs=args.preseparated_docs,
                dataset_label=base_name if args.preseparated_docs else None,
            )
            val_dataset_original = _compute_dataset_totals(val_docs)
            target_val_tokens = int(args.target_val_millions * 1_000_000)
            summary["val_total_tokens"] = val_total
            summary["val_dataset_original_tokens"] = val_dataset_original
            summary["target_val_tokens"] = target_val_tokens
            if val_total == 0:
                logger.warning("Validation split has zero tokens; nothing to subset.")
                val_scale = 0.0
            else:
                desired_val_scale = target_val_tokens / val_total
                if desired_val_scale >= 1.0:
                    logger.warning(
                        "Requested val token count (%s) exceeds available tokens (%s). Using full validation split.",
                        _format_tokens(target_val_tokens),
                        _format_tokens(val_total),
                    )
                    val_scale = 1.0
                else:
                    val_scale = desired_val_scale
            summary["val_scale"] = val_scale
            if os.path.exists(val_dest):
                shutil.rmtree(val_dest)
            actual_val_tokens, val_dataset_actual = _generate_subsets(
                val_docs,
                "val",
                val_scale,
                output_root,
                summary,
            )
            summary["actual_val_tokens"] = actual_val_tokens
            summary["val_dataset_actual_tokens"] = val_dataset_actual
            summary["val_subset"] = True
            summary["val_copied"] = False

    # Create sentence splitter for analysis
    splitter = create_token_based_sentence_splitter(
        tokenizer=tokenizer,
        use_model=False,
        max_sentence_tokens=args.max_sentence_tokens,
        min_sentence_tokens=args.min_sentence_tokens,
    )

    analysis_dir = os.path.join(output_root, "analysis")
    use_chunking = not args.no_chunking

    _run_analysis(
        output_root,
        "train",
        tokenizer,
        splitter,
        analysis_dir,
        chunk_size=args.chunk_size,
        min_sentences=args.min_sentences_per_document,
        max_sentence_tokens=args.max_sentence_tokens,
        min_sentence_tokens=args.min_sentence_tokens,
        use_chunking=use_chunking,
        cache_key="subset",
        sentence_tail_len=args.sentence_tail_len,
    )
    _run_analysis(
        output_root,
        "val",
        tokenizer,
        splitter,
        analysis_dir,
        chunk_size=args.chunk_size,
        min_sentences=args.min_sentences_per_document,
        max_sentence_tokens=args.max_sentence_tokens,
        min_sentence_tokens=args.min_sentence_tokens,
        use_chunking=use_chunking,
        cache_key="subset",
        sentence_tail_len=args.sentence_tail_len,
    )

    summary_path = os.path.join(output_root, "subset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
