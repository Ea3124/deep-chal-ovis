import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from transformers import AutoTokenizer
from ovis.model.modeling_ovis import Ovis

# ---- 이미지 디코딩 안전/속도 설정 ----
Image.MAX_IMAGE_PIXELS = None           # 초대형 이미지 경고 해제(필요시 수치로 제한 가능)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 일부 손상 이미지도 최대한 로드

def load_model(merged_dir: str):
    tok = AutoTokenizer.from_pretrained(merged_dir, use_fast=False, trust_remote_code=True)
    model = Ovis.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if getattr(model, "llm", None) is not None:
        model.llm.config.pad_token_id = tok.pad_token_id
        model.llm.config.eos_token_id = tok.eos_token_id
    else:
        model.config.pad_token_id = tok.pad_token_id
        model.config.eos_token_id = tok.eos_token_id

    return tok, model

def read_meta_all(data_dir: Path) -> List[Dict[str, Any]]:
    """new_test.json -> test.json -> (fallback) train.json + val.json"""
    fp = data_dir / "new_test.json"
    if fp.exists():
        return json.loads(fp.read_text())
    fp = data_dir / "test.json"
    if fp.exists():
        return json.loads(fp.read_text())

    metas = []
    for fn in ["train.json", "val.json"]:
        f = data_dir / fn
        if f.exists():
            metas.extend(json.loads(f.read_text()))
    return metas

def find_image_dir(data_dir: Path) -> Optional[Path]:
    di = data_dir / "datainfo.json"
    if not di.exists():
        return None
    info = json.loads(di.read_text())
    # 테스트 우선
    if "my_dataset_test" in info and "image_dir" in info["my_dataset_test"]:
        return Path(info["my_dataset_test"]["image_dir"])
    for k in ["my_dataset_train", "my_dataset_val"]:
        if k in info and "image_dir" in info[k]:
            return Path(info[k]["image_dir"])
    return None

# ---- 유틸: 배수 정렬 ----
def _round_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)

# ---- 빠른 디코딩 + 비율 유지 축소 (업스케일 방지 & 배수 정렬) ----
def _fast_open_downscale(
    pth: Path,
    max_side: int = 2048,
    min_pixels: int = 384*384,
    factor: int = 32
) -> Image.Image:
    img = Image.open(pth)
    # EXIF 회전 보정
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # 일부 포맷에서 디코딩 해상도 힌트 (효과 있을 때만 적용됨)
    try:
        img.draft("RGB", (max_side, max_side))
    except Exception:
        pass
    img.load()

    w, h = img.width, img.height
    area = max(1, w * h)

    # 1) 긴 변을 max_side 이하로 제한하는 축소 비율
    scale_maxside = min(1.0, max_side / max(w, h))

    # 2) 업스케일 방지: 선축소 결과가 min_pixels 이상이도록 하한 비율
    #    area * scale^2 >= min_pixels  =>  scale >= sqrt(min_pixels / area)
    import math
    need_scale_min = math.sqrt(min_pixels / area)
    scale_minpixels = 1.0 if need_scale_min > 1.0 else need_scale_min

    # 3) 실제 적용 스케일 = max(scale_minpixels, scale_maxside), 단 >1이면 1로(업스케일 금지)
    scale = max(scale_minpixels, scale_maxside)
    if scale > 1.0:
        scale = 1.0

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        # 4) 모델 전처리 배수(factor=32)로 정렬 → 후단에서 재리사이즈 최소화
        new_w = _round_to_multiple(new_w, factor)
        new_h = _round_to_multiple(new_h, factor)
        # 정렬 과정에서 과도하게 줄어 min_pixels 미만으로 떨어지면 한 단계 올림
        if new_w * new_h < min_pixels:
            new_w = _round_to_multiple(new_w + factor, factor)
            new_h = _round_to_multiple(new_h + factor, factor)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img.convert("RGB")

def open_images(
    image_dir: Optional[Path],
    item: Dict[str, Any],
    max_side: int,
    min_pixels: int = 384*384,
    factor: int = 32
) -> Optional[List[Image.Image]]:
    imgs = None
    if "image" in item and item["image"]:
        paths = item["image"]
        if isinstance(paths, str):
            paths = [paths]
        imgs = []
        for p in paths:
            pth = Path(p)
            if not pth.is_absolute():
                if image_dir is None:
                    continue
                pth = image_dir / p
            try:
                img = _fast_open_downscale(
                    pth,
                    max_side=max_side,
                    min_pixels=min_pixels,
                    factor=factor
                )
                imgs.append(img)
            except Exception as e:
                print(f"[WARN] failed to open image {pth}: {e}")
        if len(imgs) == 0:
            imgs = None
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="prepare_ovis_dataset_{test,new}.py 출력 디렉토리")
    ap.add_argument("--merged-dir", required=True, help="병합된(merge) 모델 디렉토리")
    ap.add_argument("--out", default="submission.csv", help="제출 파일 경로")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--min_pixels", type=int, default=384*384)
    ap.add_argument("--max_pixels", type=int, default=1024*1024)
    ap.add_argument("--open-max-side", type=int, default=2048, help="PIL 디코딩 직후 선축소 최대 변 길이")
    ap.add_argument("--open-factor", type=int, default=32, help="선축소 시 가로/세로 배수 정렬 (patch*stride)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = find_image_dir(data_dir)

    metas = read_meta_all(data_dir)
    total = len(metas)
    print(f"[INFO] preprocessed test samples: {total}")

    tok, model = load_model(args.merged_dir)

    rows = []
    for idx, item in enumerate(metas):
        sid = int(item.get("id", idx))
        # human 프롬프트만 추출
        prompt = None
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                prompt = turn.get("value","")
                break
        if prompt is None:
            prompt = ""

        imgs = open_images(
            image_dir,
            item,
            max_side=args.open_max_side,
            min_pixels=args.min_pixels,
            factor=args.open_factor
        )

        try:
            if imgs is None:
                resp, _, _ = model.chat(
                    prompt=prompt,
                    images=None,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                resp, _, _ = model.chat(
                    prompt=prompt,
                    images=imgs,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                )
        except RuntimeError as e:
            # 실패하면 재시도
            print(f"[WARN] id={sid} generation failed ({e}); retrying")
            try:
                if imgs is None:
                    resp, _, _ = model.chat(
                        prompt=prompt,
                        images=None,
                        do_sample=False,
                        max_new_tokens=2048,
                    )
                else:
                    resp, _, _ = model.chat(
                        prompt=prompt,
                        images=imgs,
                        min_pixels=max(256*256, args.min_pixels//2),
                        max_pixels=min(args.max_pixels, 768*768),
                        do_sample=False,
                        max_new_tokens=min(512, args.max_new_tokens),
                    )
            except Exception as e2:
                print(f"[ERROR] id={sid} second attempt failed: {e2}")
                resp = ""

        rows.append({"id": sid, "output": str(resp).strip()})
        if (idx+1) % 20 == 0:
            print(f"[INFO] processed {idx+1}/{total}")

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE] wrote {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
