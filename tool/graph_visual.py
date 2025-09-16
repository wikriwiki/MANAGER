#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph, k_hop_subgraph


# ─────────────────────────────────────────────
# 해시 함수: 프로젝트에서 쓰신 것과 동일
# ─────────────────────────────────────────────
def _hash(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest() + ".pt"


def _load_pyg_data(pt_path: Path) -> Data:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        # 혹시 dict 형태로 저장됐다면 흔한 키 시도
        for k in ["data", "graph", "g", "G"]:
            if k in obj and isinstance(obj[k], Data):
                return obj[k]
    raise ValueError(f"PyG Data를 {pt_path}에서 찾지 못했습니다.")


def _degree_topk_subgraph(data: Data, max_nodes: int) -> Data:
    if data.num_nodes <= max_nodes:
        return data
    G = to_networkx(data, to_undirected=True)
    deg = dict(G.degree())
    keep = sorted(deg.keys(), key=lambda n: deg[n], reverse=True)[:max_nodes]
    keep = torch.tensor(keep, dtype=torch.long)
    ei, _ = subgraph(keep, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

    sub = Data()
    sub.edge_index = ei
    # 노드 임베딩이 있다면 유지
    if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
        sub.x = data.x[keep]
    # edge_type이 있다면(순서가 달라질 수 있으므로) 생략해도 시각화엔 큰 문제 없음
    if hasattr(data, "edge_type"):
        # 단순 시각화를 위해 edge_type은 없애도 무방
        pass
    return sub


def _khop_subgraph(data: Data, center: int, khop: int) -> Data:
    subset, edge_index2, _, edge_mask = k_hop_subgraph(
        center, khop, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes, return_edge_mask=True
    )
    sub = Data()
    sub.edge_index = edge_index2
    if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
        sub.x = data.x[subset]
    if hasattr(data, "edge_type") and data.edge_type is not None and edge_mask is not None:
        try:
            sub.edge_type = data.edge_type[edge_mask]
        except Exception:
            pass
    return sub


def visualize_graph(data: Data, out_path: Path, layout: str = "spring", seed: int = 42, show_labels: bool = False):
    # PyG → NetworkX (무방향으로 그리면 보기 편함)
    G = to_networkx(data, to_undirected=True)

    # 레이아웃
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(10, 8), dpi=250)

    # 노드 그리기 (단색)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=80,
        node_color="#68a0cf",
        linewidths=0.4,
        edgecolors="black",
        alpha=0.9
    )

    # 엣지 그리기: edge_type이 있으면 타입별 색상(간단 팔레트), 없으면 회색
    edge_colors = "gray"
    if hasattr(data, "edge_type") and data.edge_type is not None and data.edge_index.size(1) == data.edge_type.numel():
        et = data.edge_type.cpu().tolist()
        uniq = sorted(set(et))
        palette = ["#6c757d", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color_map = {t: palette[i % len(palette)] for i, t in enumerate(uniq)}
        # networkx는 edge_index 순서를 모름 → 단순히 동일 색으로 그리면 섞여버림.
        # 따라서 타입별로 분리해 여러 번 그리는 방식 사용.
        # 먼저 엣지를 전체 회색으로 얇게 그리고, 타입별로 약간 두껍게 오버레이.
        nx.draw_networkx_edges(G, pos, width=0.6, edge_color="#bbbbbb", alpha=0.5)
        # 타입별 오버레이
        # data.edge_index의 순서대로 (u,v) 튜플 리스트를 만들고, 타입별로 필터링
        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        for t in uniq:
            mask = [i for i, tt in enumerate(et) if tt == t]
            edges_t = [(edges[i][0], edges[i][1]) for i in mask]
            nx.draw_networkx_edges(G, pos, edgelist=edges_t, width=1.0, edge_color=color_map[t], alpha=0.7)
        edge_colors = None  # 이미 위에서 지정했으니 기본선은 끔
    else:
        nx.draw_networkx_edges(G, pos, width=0.8, edge_color="#999999", alpha=0.6)

    # 라벨 (옵션)
    if show_labels:
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] 저장 완료: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="캐시된 비디오 그래프 시각화")
    ap.add_argument("--video-id", type=str, default="68UYBhIte3U", help="비디오 아이디")
    ap.add_argument("--cache-dir", type=str, default="cache", help="그래프 캐시 폴더")
    ap.add_argument("--out", type=str, default=None, help="출력 이미지 경로 (기본: graph_<video_id>.png)")
    ap.add_argument("--layout", type=str, default="spring", choices=["spring", "kamada", "spectral"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-labels", action="store_true", help="노드 ID 라벨 표시")
    ap.add_argument("--max-nodes", type=int, default=None, help="큰 그래프라면 상위 차수 노드만 남겨 그리기")
    ap.add_argument("--khop", type=int, default=None, help="k-hop 서브그래프")
    ap.add_argument("--center", type=int, default=None, help="k-hop 중심 노드 id")
    args = ap.parse_args()

    key = f"video_graph::{args.video_id}"
    filename = _hash(key)
    pt_path = Path(args.cache_dir) / filename
    out_path = Path(args.out) if args.out else Path(f"graph_{args.video_id}.png")

    print(f"[INFO] video_id={args.video_id}")
    print(f"[INFO] hash key='{key}'  →  file='{filename}'")
    print(f"[INFO] cache path: {pt_path}")

    if not pt_path.exists():
        raise FileNotFoundError(f"그래프 파일이 없습니다: {pt_path}\n"
                                f"※ 먼저 build_and_cache_graphs()를 실행해 캐시를 생성했는지 확인하세요.")

    data = _load_pyg_data(pt_path)

    # 서브그래프 옵션 적용
    if args.khop is not None and args.center is not None:
        data = _khop_subgraph(data, center=args.center, khop=args.khop)
    elif args.max_nodes is not None:
        data = _degree_topk_subgraph(data, max_nodes=args.max_nodes)

    visualize_graph(
        data,
        out_path=out_path,
        layout=args.layout,
        seed=args.seed,
        show_labels=args.show_labels
    )


if __name__ == "__main__":
    main()
