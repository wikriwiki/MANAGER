import os, json, hashlib, random, sqlite3
from pathlib import Path
# [BUG FIX] Python 3.9 호환성을 위해 typing 모듈에서 Optional, Set을 임포트합니다.
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm

class VideoPersonDataset(Dataset):
    """
    [수정됨] Python 3.9 호환성을 수정한 최종 데이터셋.
    """
    def __init__(
        self,
        db_path: str,
        split: str,
        cache_dir: str,
        # [BUG FIX] int | None -> Optional[int] 로 수정
        max_samples: Optional[int] = None,
        # [BUG FIX] Set[str] | None -> Optional[Set[str]] 로 수정
        filter_ids: Optional[Set[str]] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        all_videos_in_db = [row["video_id"] for row in self.conn.execute("SELECT DISTINCT video_id FROM video_metadata")]
        
        print("Verifying cache files...")
        available_videos = [vid for vid in tqdm(all_videos_in_db, desc="Verifying cache") if (self.cache_dir / self._hash(f"video_graph::{vid}")).exists()]
        print(f"Found {len(available_videos)} cached graphs out of {len(all_videos_in_db)} videos.")

        final_videos = [v for v in available_videos if filter_ids is None or str(v) in filter_ids]
        
        random.Random(seed).shuffle(final_videos)
        n = len(final_videos)
        n_train, n_val = int(0.7 * n), int(0.8 * n)
        if split == 'train': self.video_ids = final_videos[:n_train]
        elif split == 'val': self.video_ids = final_videos[n_train:n_val]
        else: self.video_ids = final_videos[n_val:]

        self.index = []
        for vid in self.video_ids:
            meta = self.conn.execute("SELECT persons_found FROM video_metadata WHERE video_id=?", (vid,)).fetchone()
            if meta and meta["persons_found"]:
                persons_data = json.loads(meta["persons_found"])
                for person, label in persons_data.items():
                    self.index.append({"video_id": vid, "person": person, "label": int(label)})
        
        if max_samples:
            self.index = self.index[:max_samples]

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest() + ".pt"

    def __getitem__(self, idx: int) -> Optional[Data]:
        meta = self.index[idx]
        video_id, person, label = meta['video_id'], meta['person'], meta['label']

        graph_path = self.cache_dir / self._hash(f"video_graph::{video_id}")
        
        try:
            graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)

            if not isinstance(graph_data, Data) or not hasattr(graph_data, 'x'):
                print(f"[ERROR] Corrupted file is not a valid Data object and will be skipped: {graph_path}")
                return None

            if graph_data.num_nodes == 0:
                return None

            if graph_data.num_edges > 0:
                num_original_edges = graph_data.num_edges
                
                idx_mask = graph_data.edge_index < graph_data.num_nodes
                idx_mask = idx_mask.all(dim=0)
                
                type_mask = (graph_data.edge_type >= -1) & (graph_data.edge_type <= 5)
                
                combined_mask = idx_mask & type_mask
                
                if not combined_mask.all():
                    graph_data.edge_index = graph_data.edge_index[:, combined_mask]
                    if hasattr(graph_data, 'edge_type') and graph_data.edge_type is not None:
                        graph_data.edge_type = graph_data.edge_type[combined_mask]
                    
                    num_repaired = num_original_edges - graph_data.num_edges
                    if num_repaired > 0:
                        print(f"\n[WARNING] Corrupted graph repaired: {graph_path}. "
                              f"Removed {num_repaired} invalid edges.")

            graph_data.label = torch.tensor([1 - label, label], dtype=torch.float)
            graph_data.person = person
            graph_data.video_id = video_id
            
            return graph_data
        
        except Exception as e:
            print(f"[ERROR] Failed to load or repair graph {graph_path}: {e}")
            return None