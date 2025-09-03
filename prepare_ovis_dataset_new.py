# prepare_ovis_dataset_new.py
import os, re, io, json, math, random, argparse, base64, mimetypes, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import pandas as pd
from PIL import Image
import requests

from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from http.client import IncompleteRead
from requests.exceptions import ChunkedEncodingError, ReadTimeout, ConnectionError

def _supports_range(headers: dict) -> bool:
    return (headers.get("Accept-Ranges","").lower() == "bytes") or ("Content-Range" in headers)

def _content_length(headers: dict) -> int | None:
    try:
        return int(headers.get("Content-Length"))
    except Exception:
        return None
    
def _download_with_resume(url: str, fp: Path, max_retries: int = 3, sleep_sec: float = 0.4) -> None:
    """
    안정적인 다운로드:
      - HEAD로 메타 확인
      - GET(stream=True)으로 읽기
      - 읽는 중 끊기면 IncompleteRead/ChunkedEncodingError 잡아서 재시도
      - 서버가 범위 요청 지원하면 Range 재개
    """
    # 0) HEAD (있으면)
    try:
        hr = _session.head(url, timeout=(5, 10), allow_redirects=True, headers=_headers_for(url))
        hr.raise_for_status()
        total_len = _content_length(hr.headers)
        range_ok = _supports_range(hr.headers)
    except Exception:
        total_len, range_ok = None, False

    # 1) 최초 시도 (full GET)
    attempt = 0
    downloaded = 0
    mode = "wb"
    headers = _headers_for(url) | {"Accept-Encoding": "identity"}  # 압축 끄면 안정적일 때 많음

    while attempt <= max_retries:
        try:
            # Range 재개
            if range_ok and downloaded > 0:
                headers = headers | {"Range": f"bytes={downloaded}-"}

            with _session.get(url, timeout=(5, 60), allow_redirects=True, stream=True, headers=headers) as r:
                r.raise_for_status()

                # Range 비지원인데 이전에 부분 파일 있다면 처음부터 다시
                if downloaded > 0 and not (range_ok or r.status_code == 206):
                    downloaded = 0
                    mode = "wb"

                with open(fp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        # 원래 길이를 알면 sanity check (선택)
                        if total_len is not None and downloaded > total_len + 1024:
                            raise ValueError("Downloaded more than Content-Length (server mismatch)")

            # 성공적으로 루프 끝났으면 리턴
            return

        except (IncompleteRead, ChunkedEncodingError, ReadTimeout, ConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            # 파일이 일부라도 내려왔으면 Range 재개를 위해 append 모드로 전환
            if fp.exists():
                downloaded = fp.stat().st_size
                mode = "ab"
            time.sleep(sleep_sec)  # 백오프

# --- 전역 세션: 재시도/백오프/커넥션 풀 설정 ---
_session = requests.Session()
_retries = Retry(
    total=3,                 # 총 재시도 횟수 (필요시 5까지 올려도 됨)
    connect=3,
    read=3,
    backoff_factor=0.6,      # 0.6, 1.2, 2.4 ... 지수 백오프
    status_forcelist=[403, 429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=_retries)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# -----------------------------
# Stronger, task-specific prompts
# -----------------------------
PROMPTS = {
    # 1) Image Captioning
    "caption": (
        "<image>\n"
        "Describe the image in detail."
    ),

    # 2) Visual Question Answering (answer only)
    "vqa": (
        "<image>\nQ: {question}\nA: "
        "Answer with the exact short answer only. "
        "Do not include explanations, reasoning, or extra words."
        "Never include explanations, reasoning, or extra words."
    ),

    # 3) Math Reasoning (must end with '\n#### <answer>')
    "math": (
        "{input}\n"
        "Solve the problem step by step and do some checking."
        "You must end your response accurately with a final line that begins with '#### ' followed by the final answer only."
        "Final line Example like this, '#### 7'."
    ),

    # 4) Summarization (as-is)
    "summarization": "Summarize the following document concisely:\n{input}",

    # 5) In-context Question Answering (JSON-only output)
    # Note: we will render {questions} as a bullet list from list/str.
    "text_qa": (
        "You are given a passage (Context) and a list of Questions.\n"
        "Your task is to answer each question ONLY based on the passage.\n"
        "Do NOT generate explanations, reasoning, or extra text.\n"
        "Return the output strictly in the following JSON format:\n\n"
        "{{\n"
        '  "input_text": [\n'
        '    "answer to question 1",\n'
        '    "answer to question 2",\n'
        "    ...\n"
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- The order of answers must exactly match the order of the questions.\n"
        "- Each answer should be short, factual, and derived only from the context.\n"
        "- Do not include any text outside the JSON object.\n\n"
        "Context:\n{input}\n\n"
        "Questions:\n{question}\n"
    ),
}

def infer_task_name(task: str) -> str:
    t = (task or "").lower()
    if "caption" in t: return "caption"
    if "vqa" in t: return "vqa"
    if "math" in t: return "math"
    if "sum"  in t: return "summarization"
    return "text_qa"

def _headers_for(url: str) -> dict:
    p = urlparse(url)
    referer = f"{p.scheme}://{p.netloc}/"
    return {
        "User-Agent": _DEFAULT_UA,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,        # 일부 서버는 리퍼러 요구
        "Connection": "keep-alive",
    }

_MAX_BYTES = 50 * 1024 * 1024  # 50MB cap

def _guess_ext_from_headers(hdrs) -> str:
    ct = (hdrs.get("Content-Type") or "").split(";")[0].strip().lower()
    # 일부 서버는 제대로 된 CT를 안줌 -> 기본 jpg
    if not ct:
        return ".jpg"
    ext = mimetypes.guess_extension(ct) or ".jpg"
    # .jpe 같은 애매한 확장자 정리
    if ext in (".jpe", ".jfif"):
        ext = ".jpg"
    return ext

import time, re, mimetypes
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 전역 세션: 재시도/백오프/커넥션 풀 설정 ---
_session = requests.Session()
_retries = Retry(
    total=3,                 # 총 재시도 횟수 (필요시 5까지 올려도 됨)
    connect=3,
    read=3,
    backoff_factor=0.6,      # 0.6, 1.2, 2.4 ... 지수 백오프
    status_forcelist=[403, 429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=_retries)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

def _headers_for(url: str) -> dict:
    p = urlparse(url)
    referer = f"{p.scheme}://{p.netloc}/"
    return {
        "User-Agent": _DEFAULT_UA,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,        # 일부 서버는 리퍼러 요구
        "Connection": "keep-alive",
    }

_MAX_BYTES = 50 * 1024 * 1024  # 50MB cap

def _guess_ext_from_headers(hdrs) -> str:
    ct = (hdrs.get("Content-Type") or "").split(";")[0].strip().lower()
    # 일부 서버는 제대로 된 CT를 안줌 -> 기본 jpg
    if not ct:
        return ".jpg"
    ext = mimetypes.guess_extension(ct) or ".jpg"
    # .jpe 같은 애매한 확장자 정리
    if ext in (".jpe", ".jfif"):
        ext = ".jpg"
    return ext

def save_image_from_any(x: Any, out_dir: Path, fname_hint: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(x, str) and x.startswith(("http://","https://")):
        url = x
        # 확장자 추정 (HEAD에서 못 얻으면 .jpg)
        try:
            hr = _session.head(url, timeout=(5,10), allow_redirects=True, headers=_headers_for(url))
            hr.raise_for_status()
            ext = _guess_ext_from_headers(hr.headers)
        except Exception:
            ext = ".jpg"

        fp = out_dir / f"{fname_hint}{ext}"
        try:
            _download_with_resume(url, fp, max_retries=3, sleep_sec=0.5)
        except Exception as e:
            # 마지막 fallback: 내용 전체를 메모리로 받아서 PIL로 열어 저장 (작은 파일에 한함)
            r = _session.get(url, timeout=(5, 60), allow_redirects=True, headers=_headers_for(url))
            r.raise_for_status()
            try:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                fp = out_dir / f"{fname_hint}.png"
                img.save(fp)
            except Exception:
                raise e

        # 형식 검증 (깨진 파일 대비)
        try:
            img = Image.open(fp)
            img.verify()  # 파일 일관성 점검
        except Exception:
            # verify 실패 시 다시 열어 재저장 시도
            img = Image.open(fp).convert("RGB")
            img.save(fp)

        time.sleep(0.15)  # 도메인 과도요청 방지
        return fp.name

    # base64
    if isinstance(x, str) and x.strip().startswith(("iVBOR","/9j/","R0lGOD")):
        img = Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
        fp = out_dir / f"{fname_hint}.png"; img.save(fp); return fp.name

    # bytes-like
    if isinstance(x, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(x))).convert("RGB")
        fp = out_dir / f"{fname_hint}.png"; img.save(fp); return fp.name

    raise ValueError("Unsupported image input format")


def row_to_conversation(row: Dict[str, Any], image_name: str|None) -> Dict[str, Any]:
    task = infer_task_name(row.get("task",""))
    input_type = (row.get("input_type") or "").lower()
    user_text = ""
    if task == "caption":
        user_text = PROMPTS["caption"]
    elif task == "vqa":
        user_text = PROMPTS["vqa"].format(question=row.get("question",""))
    elif task == "math":
        user_text = PROMPTS["math"].format(input=row.get("input",""))
    elif task == "summarization":
        user_text = PROMPTS["summarization"].format(input=row.get("input",""))
    else:  # in-context QA
        user_text = PROMPTS["text_qa"].format(input=row.get("input",""), question=row.get("question",""))

    convo = {"id": int(row.get("id", 0))}
    gpt_value = str(row.get("output","")).strip()

    # In-context QA (output에 dict 형태로 저장된 경우 처리)
    if task == "text_qa":
        try:
            parsed = eval(gpt_value) if isinstance(gpt_value, str) else gpt_value
            if isinstance(parsed, dict) and 'input_text' in parsed:
                gpt_value = {'input_text': parsed['input_text']}
        except Exception:
            pass

    if input_type == "image":
        convo["image"] = image_name
    else:
        convo["image"] = None

    convo["conversations"] = [
        {"from":"human","value": user_text},
        {"from":"gpt","value": gpt_value}
    ]
    return convo

def _normalize_images_and_image_tokens(samples):
    """
    samples: List[Dict]  # meta 리스트 (row_to_conversation 결과물)
    - 이미지가 없으면 <image> 토큰 제거하고 image 키 삭제
    - 이미지가 문자열이면 리스트로 표준화
    - <image> 태그 개수가 이미지 개수보다 많으면 초과 태그 제거
    """
    for s in samples:
        img = s.get("image", None)
        no_img = (img is None) or (isinstance(img, str) and img.strip() == "") or (isinstance(img, list) and len(img) == 0)

        if no_img:
            # (A) 이미지 없음 → <image> 토큰 제거 + image 키 삭제
            for m in s.get("conversations", []):
                if isinstance(m.get("value"), str):
                    m["value"] = re.sub(r"<image>\n?", "", m["value"])
            s.pop("image", None)
            continue

        # (B) 이미지가 문자열이면 리스트로 표준화
        if isinstance(img, str):
            s["image"] = [img]
        elif isinstance(img, list):
            # 빈 문자열 들어있는 경우 정리
            s["image"] = [x for x in img if isinstance(x, str) and x.strip()]

        # (C) <image> 태그 개수 > 이미지 개수 면, 초과 태그 제거
        text_concat = "".join(m.get("value", "") for m in s.get("conversations", []))
        tag_cnt = len(re.findall(r"<image>", text_concat))
        num_imgs = len(s.get("image", [])) if isinstance(s.get("image", None), list) else 0
        if tag_cnt > num_imgs:
            deficit = tag_cnt - num_imgs
            for m in s.get("conversations", []):
                if deficit <= 0:
                    break
                val = m.get("value")
                if isinstance(val, str) and "<image>" in val:
                    # 한 메시지에서 필요한 만큼만 순차 제거
                    while deficit > 0 and "<image>" in m["value"]:
                        m["value"] = m["value"].replace("<image>", "", 1)
                        deficit -= 1

    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image-dirname", default="images")
    ap.add_argument("--val-ratio", type=float, default=0.02)  # 2% 검증
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / args.image_dirname; img_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(args.parquet)
    rows = df.to_dict("records")

    meta: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        img_name = None
        if (r.get("input_type","").lower() == "image"):
            try:
                img_name = save_image_from_any(r.get("input"), img_dir, f"{r.get('id',i)}")
            except Exception as e:
                print(f"[WARN] image save failed at id={r.get('id',i)}: {e}")
                continue
        meta.append(row_to_conversation(r, img_name))

    # >>> 여기서 한 번에 정규화 <<<
    _normalize_images_and_image_tokens(meta)

    # train/val split
    random.shuffle(meta)
    n_val = max(1, int(len(meta)*args.val_ratio))
    val_meta = meta[:n_val]
    train_meta = meta[n_val:]

    (out_root/"train.json").write_text(json.dumps(train_meta, ensure_ascii=False))
    (out_root/"val.json").write_text(json.dumps(val_meta, ensure_ascii=False))

    datainfo = {
        "my_dataset_train": {
            "meta_file": str(out_root/"train.json"),
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_dir": str(img_dir)
        },
        "my_dataset_val": {
            "meta_file": str(out_root/"val.json"),
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_dir": str(img_dir)
        }
    }
    (out_root/"datainfo.json").write_text(json.dumps(datainfo, ensure_ascii=False, indent=2))
    print("Done.")
    print("train.json:", out_root/"train.json")
    print("val.json:",   out_root/"val.json")
    print("datainfo.json:", out_root/"datainfo.json")
    print("images dir:", img_dir)

if __name__ == "__main__":
    main()
