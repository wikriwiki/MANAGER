import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch_geometric.data import Data # 그래프 객체인지 확인하기 위해 임포트

def validate_graph_cache():
    """
    캐시 디렉토리 내의 .pt 파일 중 그래프 데이터 객체만을 대상으로
    유효성을 검사하여 손상된 파일을 찾아냅니다.
    """
    cache_dir = Path("cache/")
    if not cache_dir.exists():
        print(f"오류: 캐시 디렉토리 '{cache_dir}'를 찾을 수 없습니다.")
        return

    corrupted_files = []
    print(f"'{cache_dir}' 디렉토리에서 그래프 캐시 파일을 검사합니다...")

    # 캐시 디렉토리의 모든 .pt 파일을 순회합니다.
    file_list = list(cache_dir.glob("*.pt"))
    for f_path in tqdm(file_list, desc="Validating cache files"):
        try:
            data = torch.load(f_path, map_location="cpu", weights_only=False)

            # --- [핵심 수정] 로드한 데이터가 그래프 객체인지 먼저 확인 ---
            if isinstance(data, Data):
                # 그래프 객체인 경우에만 유효성 검사를 수행합니다.
                if data.num_nodes > 0 and data.num_edges > 0:
                    if data.edge_index.max().item() >= data.num_nodes:
                        # 엣지가 존재하지 않는 노드를 가리키는 경우
                        corrupted_files.append({
                            "path": str(f_path),
                            "reason": f"Edge index ({data.edge_index.max().item()}) is out of bounds for {data.num_nodes} nodes."
                        })
            # else:
            #   그래프 객체가 아닌 경우 (예: 텍스트, 비디오 임베딩 텐서)는 조용히 건너뜁니다.

        except Exception as e:
            # 파일을 로드하는 과정 자체에서 오류가 발생한 경우
            corrupted_files.append({
                "path": str(f_path),
                "reason": f"Failed to load file (Load Error: {e})"
            })

    if corrupted_files:
        print(f"\n[오류] 총 {len(corrupted_files)}개의 손상된 그래프 파일을 찾았습니다:")
        for f_info in corrupted_files:
            print(f" - 파일: {f_info['path']}")
            print(f"   원인: {f_info['reason']}")
        print("\n[권장 조치] 손상된 캐시 파일을 삭제하거나, cache/ 디렉토리 전체를 삭제 후")
        print("           'build_and_cache_graphs.py'를 다시 실행하여 모든 그래프를 재생성하세요.")
    else:
        print("\n[성공] 모든 그래프 캐시 파일이 정상입니다.")

if __name__ == "__main__":
    validate_graph_cache()