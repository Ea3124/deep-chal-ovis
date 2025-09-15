# 데이터 준비 & 환경설정

**Kaggle**에서 데이터 내려받아 압축 풀기

```bash
conda env create -n ovis -f environment.yml
pip install -r requirements.txt

conda activate ovis 

# 작업 루트 예시
export ROOT=<YOUR_PATH>/deep-chal-ovis
mkdir -p "$ROOT/dataset" && cd "$ROOT/dataset"

# 다운로드 & 압축해제
kaggle competitions download -c deeplearningchallenge
unzip deeplearningchallenge.zip
```

# 학습(Training)

1. **학습용 JSON 생성**

```bash
cd $ROOT
python prepare_ovis_dataset_new.py \
  --parquet "$ROOT/dataset/deep_chal_multitask_dataset.parquet" \
  --outdir "$ROOT/train_data" \
  --image-dirname images \
  --val-ratio 0.02 \
  --seed 42
```

2. **datainfo.json 경로 수정**
   파일: `$ROOT/third_party/ovis/ovis/train/dataset/datainfo.json`

```json
{
  "my_dataset_train": {
    "meta_file": "<YOUR_PATH>/deep-chal-ovis/train_data/train.json",
    "storage_type": "hybrid",
    "data_format": "conversation",
    "image_dir": "<YOUR_PATH>/deep-chal-ovis/train_data/images"
  },
  "my_dataset_val": {
    "meta_file": "<YOUR_PATH>/deep-chal-ovis/train_data/val.json",
    "storage_type": "hybrid",
    "data_format": "conversation",
    "image_dir": "<YOUR_PATH>/deep-chal-ovis/train_data/images"
  }
}
```

3. **학습 실행**

> 스크립트가 상대경로를 쓰므로 **반드시 서브모듈 루트에서 실행**

```bash
cd $ROOT/third_party/ovis
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_ovis2_5_lora_mt_2.sh
```

## Merged lora in base 추가 예정

# 추론(Inference)

1. **체크포인트 다운로드(huggingface load 실패시에만 시도)**

   ([huggingface 다운로드 링크](https://huggingface.co/ea3124/azu2025) → 아래처럼 모델 디렉터리 생성 후 배치)

2. **체크포인트 배치 경로 예시**

```
$ROOT/third_party/ovis/ovis/checkpoints/
└── run_ovis2_5_lora_mt_2_merged
    ├── added_tokens.json
    ├── config.json
    ├── merges.txt
    ├── model-00001-of-00005.safetensors
    ├── model-00002-of-00005.safetensors
    ├── model-00003-of-00005.safetensors
    ├── model-00004-of-00005.safetensors
    ├── model-00005-of-00005.safetensors
    ├── model.safetensors.index.json
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
```

3. **테스트용 JSON 생성**

```bash
cd $ROOT
python prepare_ovis_dataset_test.py \
  --parquet "$ROOT/dataset/deep_chal_multitask_dataset_test.parquet" \
  --outdir "$ROOT/test_data"

# 그 후 id 재지정
cd $ROOT/deep-chal-ovis/test_data
python re_index.py
```

4. **datainfo.json 경로 수정**
   파일: `$ROOT/test_data/datainfo.json`

```json
{
  "my_dataset_test": {
    "meta_file": "$ROOT/test_data/test_reindexed.json",
    "storage_type": "hybrid",
    "data_format": "conversation",
    "image_dir": "/home/undergrad/deep-chal-ovis/test_data/images"
  }
}
```

4. **추론 실행 → submission.csv 생성**

```bash
cd $ROOT
export CUDA_VISIBLE_DEVICES=4,5,6,7   # 필요 시 변경

# huggingface 모델 사용 될시
python infer_submit.py \
  --data-dir /home/undergrad/deep-chal-ovis/test_data \
  --model-id ea3124/azu2025 \
  --out submission.csv \
  --open-max-side 1536 \
  --min_pixels $((448*448)) \
  --max_pixels $((1344*1792))


# huggingface 모델 사용이 안되어 수동 다운로드 받았을 시
python infer_submit.py \
  --data-dir "$ROOT/test_data" \
  --merged-dir /home/undergrad/ovis/Ovis/checkpoints/run_ovis2_5_lora_mt_2_merged \
  --out submission.csv \
  --open-max-side 1536 \
  --min_pixels $((448*448)) \
  --max_pixels $((1344*1792))
```

