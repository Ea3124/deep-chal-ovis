
import argparse, json, random
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# 기존 전처리 유틸 재사용
from prepare_ovis_dataset_new import (
    row_to_conversation,
    _normalize_images_and_image_tokens,
    save_image_from_any,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image-dirname", default="images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / args.image_dirname; img_dir.mkdir(parents=True, exist_ok=True)

    # parquet 로드
    df = pd.read_parquet(args.parquet)
    rows = df.to_dict("records")

    # meta 구성 (이미지 저장 포함)
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

    # 이미지/<image> 토큰 정합성 보정
    meta = _normalize_images_and_image_tokens(meta)

    # 테스트 전용 단일 json으로 저장
    (out_root/"test.json").write_text(json.dumps(meta, ensure_ascii=False))

    # datainfo: 테스트 전용 키만 생성
    datainfo = {
        "my_dataset_test": {
            "meta_file": str(out_root/"test.json"),
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_dir": str(img_dir)
        }
    }
    (out_root/"datainfo.json").write_text(json.dumps(datainfo, ensure_ascii=False, indent=2))

    # 총 개수 반환(프린트 & 파일 저장)
    total = len(meta)
    (out_root/"count.txt").write_text(str(total))
    print(f"[DONE] test.json samples: {total}")
    print("test.json:", out_root/"test.json")
    print("datainfo.json:", out_root/"datainfo.json")
    print("images dir:", img_dir)

if __name__ == "__main__":
    main()
