# validate_cache.py
import torch
from pathlib import Path
from tqdm.auto import tqdm

cache_dir = Path("cache/")
corrupted_files = []

for f_path in tqdm(list(cache_dir.glob("*.pt"))):
    try:
        data = torch.load(f_path, map_location="cpu", weights_only=False)
        if data.num_nodes > 0 and data.edge_index.max().item() >= data.num_nodes:
            corrupted_files.append(str(f_path))
    except Exception as e:
        corrupted_files.append(f"{f_path} (Load Error: {e})")

if corrupted_files:
    print(f"\n총 {len(corrupted_files)}개의 손상된 파일을 찾았습니다:")
    for f in corrupted_files:
        print(f" - {f}")
else:
    print("\n모든 캐시 파일이 정상입니다.")