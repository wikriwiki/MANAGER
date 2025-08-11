# import pandas as pd
# import sqlite3
# import torch
# import hashlib
# import json
# import re
# from functools import lru_cache
# from pathlib import Path
# from typing import Dict, List, Tuple
# from torch_geometric.data import Data
# from transformers import BertTokenizerFast, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch.nn as nn
# import os
# import sys

# # GPU 사용 여부 확인 및 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 해시 함수 정의
# def _hash(key: str) -> str:
#     return hashlib.md5(key.encode()).hexdigest() + ".pt"

# # Tensor를 지정된 디바이스로 옮기는 헬퍼 함수
# def _to_device(data: Dict, device: torch.device) -> Dict:
#     return {k: v.to(device) for k, v in data.items()}

# # VRAM 로깅 함수
# def log_vram(stage: str, device: torch.device):
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#         dev_id = device.index if isinstance(device, torch.device) else torch.cuda.current_device()
#         alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
#         reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
#         peak_a = torch.cuda.max_memory_allocated(dev_id) / 1024**2
#         peak_r = torch.cuda.max_memory_reserved(dev_id) / 1024**2
#         wasted = reserved - alloc
#         print(f"[{stage:15s}] alloc: {alloc:6.1f} MB | reserved: {reserved:6.1f} MB | "
#               f"peak_alloc: {peak_a:6.1f} MB | peak_reserved: {peak_r:6.1f} MB | wasted: {wasted:6.1f} MB")
#         torch.cuda.reset_peak_memory_stats(dev_id)

# # Edge type id 정의
# EDGE_TYPE: Dict[str, int] = {
#     "t_t": 0,  # text  ↔ text
#     "v_v": 1,  # video ↔ video
#     "a_a": 2,  # audio ↔ audio
#     "t_v": 3,  # text  ↔ video
#     "t_a": 4,  # text  ↔ audio
#     "utt": 5,  # utterance ↔ utterance (for merging)
# }

# # 양방향 엣지 추가 헬퍼
# def _add_bidir(
#     src: int,
#     dst: int,
#     etype: int,
#     edge_src: List[int],
#     edge_dst: List[int],
#     edge_type: List[int],
# ):
#     edge_src.extend([src, dst])
#     edge_dst.extend([dst, src])
#     edge_type.extend([etype, etype])
    
# # models/graph_builder.py의 의존성 해결을 위해 필요한 클래스들을 여기에 포함
# class ExternalFinancialKnowledgeModel:
#     def __init__(self,
#                  wiki_db="data/wikidata_revisions.db",
#                  speech_db="data/speech_segments.db"):
#         self.wiki = sqlite3.connect(wiki_db); self.wiki.row_factory = sqlite3.Row
#         self.speech = sqlite3.connect(speech_db); self.speech.row_factory = sqlite3.Row

#         self.target_entities = self._load_target_entities()
#         label_df = pd.read_sql("SELECT DISTINCT qid,value AS label FROM labels "
#                                "WHERE lang='en';", self.wiki)
#         label_df = label_df[label_df["label"].str.lower()
#                             .isin([n.lower() for n in self.target_entities])]
#         label_df["id_int"] = pd.factorize(label_df["qid"])[0]

#         self.qid2int = dict(zip(label_df["qid"], label_df["id_int"]))
#         self.int2qid = {v: k for k, v in self.qid2int.items()}  # 수정된 부분 반영
#         self.label2qid = {l.lower(): q for q, l in zip(label_df["qid"],
#                                                      label_df["label"])}

#         safe = [re.sub(r"\s+", r"\\s+", re.escape(n.lower()))
#                 for n in self.target_entities]
#         self._pat = re.compile(r"(" + "|".join(safe) + r")", re.I)

#     def _load_target_entities(self) -> List[str]:
#         df = pd.read_sql("SELECT persons_found FROM video_metadata;", self.speech)
#         names = set()
#         for js in df["persons_found"]:
#             if js: names.update(json.loads(js).keys())
#         return list(names)

#     def identify_entities(self, text: str) -> List[str]:
#         t = re.sub(r"[^\w\s]", "", text.lower())
#         return list({m.strip() for m in self._pat.findall(t)})

#     def entities_to_id(self, ents: List[str]) -> List[int]:
#         return [self.qid2int[self.label2qid[e.lower()]]
#                 for e in ents if e.lower() in self.label2qid]
    
#     @lru_cache(maxsize=32)
#     def _graph_until(self, time_iso: str) -> Data:
#         sql = ("SELECT c.qid subj,c.property pid,c.value_qid obj "
#                "FROM claims c JOIN revisions r USING(qid,revision_id) "
#                "WHERE r.timestamp<=?")
#         df = pd.read_sql(sql, self.wiki, params=(time_iso,))
#         df = df[df["subj"].isin(self.qid2int) & df["obj"].isin(self.qid2int)]
#         if df.empty: return Data()

#         src = torch.tensor(df["subj"].map(self.qid2int).to_numpy(), dtype=torch.long)
#         dst = torch.tensor(df["obj"].map(self.qid2int).to_numpy(), dtype=torch.long)
#         rel = torch.tensor(pd.factorize(df["pid"])[0], dtype=torch.long).view(-1, 1)
#         return Data(edge_index=torch.stack([src, dst]), edge_attr=rel)
        
#     def acquire_related_external_knowledge(
#         self, text: str, time_iso: str,
#         add_reverse=True, add_self_loop=True
#     ) -> Tuple[List[int], Data]:
#         ids = self.entities_to_id(self.identify_entities(text))
#         G = self._graph_until(time_iso)
#         if not ids or G.edge_index.numel() == 0: return ids, Data()

#         mask = (torch.isin(G.edge_index[0], torch.tensor(ids)) |
#                 torch.isin(G.edge_index[1], torch.tensor(ids)))
#         ei, ea = G.edge_index[:, mask], G.edge_attr[mask]

#         if add_reverse:
#             ei = torch.cat([ei, ei.flip(0)], 1)
#             ea = torch.cat([ea, ea], 0)
#         if add_self_loop:
#             loops = torch.tensor(ids, dtype=torch.long, device=ei.device)
#             ei = torch.cat([ei, loops.unsqueeze(0).repeat(2, 1)], 1)
#             ea = torch.cat([ea, torch.full((len(loops), 1), -1,
#                                          dtype=torch.long, device=ei.device)], 0)
#         return ids, Data(edge_index=ei, edge_attr=ea)

# # TextFeatureExtractor 클래스 (수정된 내용 반영)
# def build_cross_map(wp_offsets: List[Tuple[int, int]],
#                     glm_offsets: List[Tuple[int, int]]) -> List[List[int]]:
#     mapping = [[] for _ in wp_offsets]; p = 0
#     for i,(ws,we) in enumerate(wp_offsets):
#         while p < len(glm_offsets) and glm_offsets[p][1] <= ws: p += 1
#         q = p
#         while q < len(glm_offsets) and glm_offsets[q][0] < we:
#             mapping[i].append(q); q += 1
#     return mapping

# class TextFeatureExtractor:
#     def __init__(self):
#         self.wp_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
#         self.glm_tok: AutoTokenizer = AutoTokenizer.from_pretrained(
#             "THUDM/chatglm2-6b", trust_remote_code=True
#         )
#         self.fast_offset = True
#         quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#         self.glm = AutoModelForCausalLM.from_pretrained(
#             "THUDM/chatglm2-6b",
#             trust_remote_code=True,
#             quantization_config=quantization_config,
#             output_hidden_states=True,
#             device_map="auto"
#         )
#         self.proj_down = nn.Linear(4096, 768, bias=False).to(device).half()
#         self.proj_up = nn.Linear(768, 4096, bias=False).to(device).half()
#         self.norm = nn.LayerNorm(768).to(device).half()

#         for m in (self.proj_down, self.proj_up):
#             nn.init.xavier_uniform_(m.weight)

#     def _manual_offsets(self, text: str, toks: List[str]):
#         norm = text.lower(); p, off = 0, []
#         for t in toks:
#             tc = t.lstrip(" "); tc = tc if tc else " "
#             j = norm.find(tc, p)
#             j = j if j != -1 else p
#             off.append((j, j + len(tc))); p = j + len(tc)
#         return off

#     @torch.no_grad()
#     def encode(self,
#                utterance_text: str,
#                knowledge_triples: list | None = None,
#                anchor_entities: list | None = None):
#         know_parts = []
#         if anchor_entities:
#             know_parts.append(" [ENT] ".join(map(str, anchor_entities)))
#         if knowledge_triples:
#             know_parts += [f"{h} [R] {r} [T] {t}" for h, r, t in knowledge_triples]
#         knowledge_text = " ".join(know_parts) if know_parts else None

#         wp = self.wp_tok(
#             utterance_text,
#             text_pair=knowledge_text,
#             return_offsets_mapping=True,
#             return_token_type_ids=True,
#             return_tensors="pt",
#         )
#         wp = _to_device(wp, self.glm.device)
#         wp_tokens = self.wp_tok.convert_ids_to_tokens(wp["input_ids"][0])
#         wp_offsets = wp["offset_mapping"][0].tolist()
#         token_types = wp["token_type_ids"][0].tolist()
#         sep_idx = [i for i, tok in enumerate(wp_tokens) if tok == "[SEP]"]

#         knowledge_blocks: List[Tuple[int, int]] = []
#         in_block, start = False, None
#         for i, (tt, tok) in enumerate(zip(token_types, wp_tokens)):
#             if tt == 1 and tok != "[SEP]":
#                 if not in_block:
#                     start, in_block = i, True
#             else:
#                 if in_block:
#                     knowledge_blocks.append((start, i))
#                     in_block = False
#         if in_block:
#             knowledge_blocks.append((start, len(wp_tokens)))

#         merged = utterance_text + (" [SEP] " + knowledge_text if knowledge_text else "")
#         glm_ids = self.glm_tok.encode(merged, add_special_tokens=False)
#         glm_enc = {
#             "input_ids": torch.tensor([glm_ids], device=self.glm.device),
#             "attention_mask": torch.ones(
#                 1, len(glm_ids), dtype=torch.long, device=self.glm.device)
#         }
#         glm_tokens = self.glm_tok.convert_ids_to_tokens(glm_ids)
#         glm_offsets = (self.glm_tok(
#             merged, add_special_tokens=False, return_offsets_mapping=True
#         )["offset_mapping"]
#                        if hasattr(self.glm_tok, "get_offsets_mapping")
#                        else self._manual_offsets(merged, glm_tokens))

#         hidden4096 = self.glm(**glm_enc).hidden_states[-1][0]
#         hid768 = self.norm(self.proj_down(hidden4096))

#         map_wp2glm = build_cross_map(wp_offsets, glm_offsets)
#         max_idx = hid768.size(0)
#         wp_emb = torch.stack([
#             (
#                 hid768[torch.tensor(valid, dtype=torch.long,
#                                     device=hid768.device)].mean(0)
#                 if (valid := [i for i in ids if i < max_idx])
#                 else torch.zeros(768, device=hid768.device)
#             )
#             for ids in map_wp2glm
#         ])

#         meta = {
#             "wp_tokens": wp_tokens,
#             "glm_tokens": glm_tokens,
#             "map_wp2glm": map_wp2glm,
#             "sep": sep_idx,
#             "knowledge_blocks": knowledge_blocks,
#         }
#         return wp_emb, meta

#     def cleanup(self):
#         print("TextFeatureExtractor 메모리 해제 중...")
#         if hasattr(self, 'glm'):
#             del self.glm
#         if hasattr(self, 'proj_down'):
#             del self.proj_down
#         if hasattr(self, 'proj_up'):
#             del self.proj_up
#         torch.cuda.empty_cache()
#         print("TextFeatureExtractor 메모리 해제 완료")
# # GraphBuilder 클래스
# class GraphBuilder:
#     """단일 발화 그래프 + 여러 발화 병합 유틸 (***양방향 엣지 버전***)"""

#     def __init__(self, time_iso: str, *, merge_anchor: bool = False):
#         self.time_iso = time_iso
#         self.merge_anchor = merge_anchor
#         self.text_enc = TextFeatureExtractor()
#         self.ekm = ExternalFinancialKnowledgeModel()

#     def __del__(self):
#         if hasattr(self, 'text_enc'):
#             self.text_enc.cleanup()
#             del self.text_enc
#         print("GraphBuilder 메모리 해제 완료")
#         log_vram("del", device)

#     def _triple_to_string(self, subj: str, pid: str, obj: str) -> str:
#         if not hasattr(self, "_pid_cache"):
#             self._pid_cache: Dict[str, str] = {}
#         if pid not in self._pid_cache:
#             row = self.ekm.wiki.execute(
#                 "SELECT label FROM property_labels WHERE pid=? LIMIT 1", (pid,)
#             ).fetchone()
#             self._pid_cache[pid] = row[0] if row else pid
#         return f"{subj} [REL] {self._pid_cache[pid].replace(' ', '_')} [REL] {obj}"

#     def build(
#         self,
#         utterance_text: str,
#         video_emb: torch.Tensor | None = None,
#         audio_emb: torch.Tensor | None = None,
#     ) -> Data:
#         """발화(텍스트+모달) → PyG Data (모든 엣지 양방향)"""
#         if video_emb is not None and video_emb.is_cuda:
#             video_emb = video_emb.cpu()
#         if audio_emb is not None and audio_emb.is_cuda:
#             audio_emb = audio_emb.cpu()

#         # 1) 외부 지식 트리플 수집
#         ent_ids, ek_sub = self.ekm.acquire_related_external_knowledge(
#             utterance_text, self.time_iso
#         )
#         triples: List[Tuple[str, str, str]] = []
#         if ek_sub.num_edges:
#             for s, d, (prop,) in zip(*ek_sub.edge_index, ek_sub.edge_attr):
#                 triples.append(
#                     (
#                         self.ekm.int2qid[s.item()],
#                         f"P{prop.item()}",
#                         self.ekm.int2qid[d.item()],
#                     )
#                 )

#         # 2) 텍스트 인코딩
#         wp_emb, meta = self.text_enc.encode(
#             utterance_text,
#             knowledge_triples=[self._triple_to_string(*t) for t in triples],
#             anchor_entities=[self.ekm.int2qid[i] for i in ent_ids],
#         )
#         if wp_emb.is_cuda:
#             wp_emb = wp_emb.cpu()
#         hs = wp_emb
#         D = hs.size(1)
#         sep0 = meta["sep"][0] if meta["sep"] else -1

#         node_feats: List[torch.Tensor] = []
#         edge_src: List[int] = []
#         edge_dst: List[int] = []
#         edge_type: List[int] = []

#         # --- utterance text 토큰 노드 ---
#         text_token_map: Dict[int, int] = {
#             idx: len(node_feats) for idx in range(1, sep0)
#         }
#         node_feats.extend([hs[i] for i in range(1, sep0)])
#         text_nodes = list(text_token_map.values())
#         for i in range(len(text_nodes) - 1):
#             _add_bidir(
#                 text_nodes[i], text_nodes[i + 1], EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type
#             )

#         # --- knowledge 토큰 노드 ---
#         knowledge_token_map: Dict[int, int] = {}
#         for s, e in meta["knowledge_blocks"]:
#             prev = None
#             for idx in range(s, e):
#                 g = len(node_feats)
#                 knowledge_token_map[idx] = g
#                 node_feats.append(hs[idx])
#                 if prev is not None:
#                     _add_bidir(prev, g, EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type)
#                 prev = g

#         # 비디오 노드
#         if video_emb is not None and video_emb.numel():
#             v_idx = len(node_feats)
#             node_feats.append(video_emb.squeeze(0))
#             for t in text_nodes:
#                 _add_bidir(t, v_idx, EDGE_TYPE["t_v"], edge_src, edge_dst, edge_type)
#         else:
#             v_idx = -1

#         # 오디오 노드
#         if audio_emb is not None and audio_emb.numel():
#             a_idx = len(node_feats)
#             node_feats.append(audio_emb.squeeze(0))
#             for t in text_nodes:
#                 _add_bidir(t, a_idx, EDGE_TYPE["t_a"], edge_src, edge_dst, edge_type)
#         else:
#             a_idx = -1

#         x = torch.stack(node_feats) if node_feats else torch.empty(0, D)
#         edge_index = (
#             torch.tensor([edge_src, edge_dst], dtype=torch.long)
#             if edge_src
#             else torch.empty(2, 0, dtype=torch.long)
#         )
#         edge_type_t = (
#             torch.tensor(edge_type, dtype=torch.long)
#             if edge_type
#             else torch.empty(0, dtype=torch.long)
#         )

#         data = Data(x=x, edge_index=edge_index, edge_type=edge_type_t)
#         data.node_meta = {
#             "text_nodes": len(text_nodes),
#             "knowledge_nodes": len(knowledge_token_map),
#             "video_nodes": 1 if v_idx != -1 else 0,
#             "audio_nodes": 1 if a_idx != -1 else 0,
#             "triples": len(triples),
#         }
#         data.utt_meta = {
#             "first_text_node": text_nodes[0] if text_nodes else -1,
#             "last_text_node": text_nodes[-1] if text_nodes else -1,
#         }
#         return data

# # models/graph_utils.py의 merge_graph 메소드
# def merge_graph(prev_graph: Data | None, current_graph: Data) -> Data:
#     if prev_graph is None or prev_graph.x.numel() == 0:
#         return current_graph

#     # 노드 피처 병합
#     x_merged = torch.cat([prev_graph.x, current_graph.x], dim=0)

#     # 엣지 인덱스 및 타입 병합 (오프셋 적용)
#     num_prev_nodes = prev_graph.x.size(0)
#     edge_index_current_offset = current_graph.edge_index + num_prev_nodes
#     edge_index_merged = torch.cat([prev_graph.edge_index, edge_index_current_offset], dim=1)

#     edge_type_merged = torch.cat([prev_graph.edge_type, current_graph.edge_type])

#     # utterance 연결 엣지 추가 (양방향)
#     if prev_graph.utt_meta["last_text_node"] != -1 and current_graph.utt_meta["first_text_node"] != -1:
#         prev_last_node = prev_graph.utt_meta["last_text_node"]
#         curr_first_node = current_graph.utt_meta["first_text_node"] + num_prev_nodes
        
#         utt_edge_index = torch.tensor([[prev_last_node, curr_first_node],
#                                        [curr_first_node, prev_last_node]], dtype=torch.long)
#         utt_edge_type = torch.tensor([EDGE_TYPE["utt"], EDGE_TYPE["utt"]], dtype=torch.long)

#         edge_index_merged = torch.cat([edge_index_merged, utt_edge_index], dim=1)
#         edge_type_merged = torch.cat([edge_type_merged, utt_edge_type])

#     # 메타 정보 병합
#     node_meta_merged = {k: prev_graph.node_meta.get(k, 0) + current_graph.node_meta.get(k, 0)
#                         for k in set(prev_graph.node_meta.keys()) | set(current_graph.node_meta.keys())}
    
#     utt_meta_merged = {
#         "first_text_node": prev_graph.utt_meta["first_text_node"],
#         "last_text_node": current_graph.utt_meta["last_text_node"] + num_prev_nodes
#     }

#     merged_graph = Data(x=x_merged, edge_index=edge_index_merged, edge_type=edge_type_merged)
#     merged_graph.node_meta = node_meta_merged
#     merged_graph.utt_meta = utt_meta_merged
#     return merged_graph

# def build_and_cache_graphs():
#     """
#     ready_videos.csv에 있는 video_id에 대해 그래프를 구축하고 캐시로 저장합니다.
#     """
#     # 파일 경로 설정
#     ready_videos_path = "data/ready_videos.csv"
#     speech_db_path = "data/speech_segments.db"
#     cache_dir = Path("cache")
#     cache_dir.mkdir(exist_ok=True)

#     # 1. ready_videos.csv에서 video_id 목록 불러오기
#     try:
#         ready_videos_df = pd.read_csv(ready_videos_path)
#         video_ids = ready_videos_df["video_id"].tolist()
#         print(f"총 {len(video_ids)}개의 video_id를 불러왔습니다.")
#     except FileNotFoundError:
#         print(f"오류: {ready_videos_path} 파일을 찾을 수 없습니다.")
#         return

#     # 2. speech_segments.db 연결
#     try:
#         conn = sqlite3.connect(speech_db_path)
#         conn.row_factory = sqlite3.Row
#         print(f"{speech_db_path}에 연결되었습니다.")
#     except sqlite3.Error as e:
#         print(f"오류: {speech_db_path} 연결 실패 - {e}")
#         return

#     # 3. 각 video_id에 대해 그래프 구축 및 캐시 저장
#     for video_id in video_ids:
#         print(f"\n[Video ID: {video_id}] 그래프 구축 시작...")
        
#         # 비디오 업로드 시간 조회
#         try:
#             video_meta = conn.execute(
#                 "SELECT published_date FROM video_metadata WHERE video_id = ?",
#                 (video_id,)
#             ).fetchone()
#             if not video_meta:
#                 print(f"경고: video_id {video_id}에 대한 메타데이터를 찾을 수 없습니다. 건너뜁니다.")
#                 continue
#             upload_time = video_meta["published_date"]
#         except sqlite3.Error as e:
#             print(f"오류: video_id {video_id}의 업로드 시간 조회 실패 - {e}")
#             continue

#         # GraphBuilder 인스턴스 생성
#         try:
#             graph_builder = GraphBuilder(time_iso=upload_time)
#         except Exception as e:
#             print(f"오류: GraphBuilder 인스턴스 생성 실패 - {e}")
#             continue

#         video_graph = None

#         # 비디오에 대한 발화 불러오기
#         speech_segments = conn.execute(
#             "SELECT segment_id, script FROM speech_segments WHERE video_id = ? ORDER BY start_time",
#             (video_id,)
#         ).fetchall()
        
#         if not speech_segments:
#             print(f"경고: video_id {video_id}에 대한 발화가 없습니다. 그래프를 생성하지 않습니다.")
#             continue
        
#         for segment in speech_segments:
#             seg_id = segment["segment_id"]
#             utterance_text = segment["script"]
#             print(f" - 발화 '{seg_id}' 처리 중...")

#             # 캐시 파일 경로 설정
#             video_key = f"vid::{seg_id}"
#             audio_key = f"aud::{seg_id}"
#             video_cache_path = cache_dir / _hash(video_key)
#             audio_cache_path = cache_dir / _hash(audio_key)

#             # 임베딩 텐서 로드
#             video_emb, audio_emb = None, None
#             try:
#                 if video_cache_path.exists():
#                     video_emb = torch.load(video_cache_path)
#                 if audio_cache_path.exists():
#                     audio_emb = torch.load(audio_cache_path)
#             except Exception as e:
#                 print(f"오류: 발화 '{seg_id}'의 캐시 파일 로드 실패 - {e}")
#                 continue

#             # 단일 발화 그래프 구축
#             try:
#                 current_graph = graph_builder.build(
#                     utterance_text=utterance_text,
#                     video_emb=video_emb,
#                     audio_emb=audio_emb
#                 )
#             except Exception as e:
#                 print(f"오류: 발화 '{seg_id}'에 대한 그래프 구축 실패 - {e}")
#                 continue

#             # 이전 그래프와 병합
#             video_graph = merge_graph(video_graph, current_graph)

#         # 전체 비디오 그래프 캐시 저장
#         if video_graph:
#             output_path = cache_dir / _hash(f"video_graph::{video_id}")
#             torch.save(video_graph, output_path)
#             print(f"\n성공: video_id {video_id}에 대한 최종 그래프가 {output_path}에 저장되었습니다.")
        
#         # GraphBuilder 인스턴스 명시적으로 삭제하여 메모리 해제 유도
#         del graph_builder

#     conn.close()
#     print("\n모든 작업이 완료되었습니다.")

# if __name__ == "__main__":
#     build_and_cache_graphs()

import pandas as pd
import sqlite3
import torch
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
from torch_geometric.data import Data
from transformers import BertTokenizerFast, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn as nn
import os
import sys

# GPU 사용 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 해시 함수 정의
def _hash(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest() + ".pt"

# Tensor를 지정된 디바이스로 옮기는 헬퍼 함수
def _to_device(data: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) for k, v in data.items()}

# VRAM 로깅 함수
def log_vram(stage: str, device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        dev_id = device.index if isinstance(device, torch.device) else torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
        reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
        peak_a = torch.cuda.max_memory_allocated(dev_id) / 1024**2
        peak_r = torch.cuda.max_memory_reserved(dev_id) / 1024**2
        wasted = reserved - alloc
        print(f"[{stage:15s}] alloc: {alloc:6.1f} MB | reserved: {reserved:6.1f} MB | "
              f"peak_alloc: {peak_a:6.1f} MB | peak_reserved: {peak_r:6.1f} MB | wasted: {wasted:6.1f} MB")
        torch.cuda.reset_peak_memory_stats(dev_id)

# Edge type id 정의
EDGE_TYPE: Dict[str, int] = {
    "t_t": 0,  # text  ↔ text
    "v_v": 1,  # video ↔ video
    "a_a": 2,  # audio ↔ audio
    "t_v": 3,  # text  ↔ video
    "t_a": 4,  # text  ↔ audio
    "utt": 5,  # utterance ↔ utterance (for merging)
}

# 양방향 엣지 추가 헬퍼
def _add_bidir(
    src: int,
    dst: int,
    etype: int,
    edge_src: List[int],
    edge_dst: List[int],
    edge_type: List[int],
):
    edge_src.extend([src, dst])
    edge_dst.extend([dst, src])
    edge_type.extend([etype, etype])
    
# models/graph_builder.py의 의존성 해결을 위해 필요한 클래스들을 여기에 포함
class ExternalFinancialKnowledgeModel:
    def __init__(self,
                 wiki_db="data/wikidata_revisions.db",
                 speech_db="data/speech_segments.db"):
        self.wiki = sqlite3.connect(wiki_db); self.wiki.row_factory = sqlite3.Row
        self.speech = sqlite3.connect(speech_db); self.speech.row_factory = sqlite3.Row

        self.target_entities = self._load_target_entities()
        label_df = pd.read_sql("SELECT DISTINCT qid,value AS label FROM labels "
                               "WHERE lang='en';", self.wiki)
        label_df = label_df[label_df["label"].str.lower()
                               .isin([n.lower() for n in self.target_entities])]
        label_df["id_int"] = pd.factorize(label_df["qid"])[0]

        self.qid2int = dict(zip(label_df["qid"], label_df["id_int"]))
        self.int2qid = {v: k for k, v in self.qid2int.items()}  # 수정된 부분 반영
        self.label2qid = {l.lower(): q for q, l in zip(label_df["qid"],
                                                     label_df["label"])}

        safe = [re.sub(r"\s+", r"\\s+", re.escape(n.lower()))
                for n in self.target_entities]
        self._pat = re.compile(r"(" + "|".join(safe) + r")", re.I)

    def _load_target_entities(self) -> List[str]:
        df = pd.read_sql("SELECT persons_found FROM video_metadata;", self.speech)
        names = set()
        for js in df["persons_found"]:
            if js: names.update(json.loads(js).keys())
        return list(names)

    def identify_entities(self, text: str) -> List[str]:
        t = re.sub(r"[^\w\s]", "", text.lower())
        return list({m.strip() for m in self._pat.findall(t)})

    def entities_to_id(self, ents: List[str]) -> List[int]:
        return [self.qid2int[self.label2qid[e.lower()]]
                for e in ents if e.lower() in self.label2qid]
    
    @lru_cache(maxsize=32)
    def _graph_until(self, time_iso: str) -> Data:
        sql = ("SELECT c.qid subj,c.property pid,c.value_qid obj "
               "FROM claims c JOIN revisions r USING(qid,revision_id) "
               "WHERE r.timestamp<=?")
        df = pd.read_sql(sql, self.wiki, params=(time_iso,))
        df = df[df["subj"].isin(self.qid2int) & df["obj"].isin(self.qid2int)]
        if df.empty: return Data()

        src = torch.tensor(df["subj"].map(self.qid2int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(df["obj"].map(self.qid2int).to_numpy(), dtype=torch.long)
        rel = torch.tensor(pd.factorize(df["pid"])[0], dtype=torch.long).view(-1, 1)
        return Data(edge_index=torch.stack([src, dst]), edge_attr=rel)
        
    def acquire_related_external_knowledge(
        self, text: str, time_iso: str,
        add_reverse=True, add_self_loop=True
    ) -> Tuple[List[int], Data]:
        ids = self.entities_to_id(self.identify_entities(text))
        G = self._graph_until(time_iso)
        if not ids or G.edge_index.numel() == 0: return ids, Data()

        mask = (torch.isin(G.edge_index[0], torch.tensor(ids)) |
                torch.isin(G.edge_index[1], torch.tensor(ids)))
        ei, ea = G.edge_index[:, mask], G.edge_attr[mask]

        if add_reverse:
            ei = torch.cat([ei, ei.flip(0)], 1)
            ea = torch.cat([ea, ea], 0)
        if add_self_loop:
            loops = torch.tensor(ids, dtype=torch.long, device=ei.device)
            ei = torch.cat([ei, loops.unsqueeze(0).repeat(2, 1)], 1)
            ea = torch.cat([ea, torch.full((len(loops), 1), -1,
                                           dtype=torch.long, device=ei.device)], 0)
        return ids, Data(edge_index=ei, edge_attr=ea)

# TextFeatureExtractor 클래스 (수정된 내용 반영)
def build_cross_map(wp_offsets: List[Tuple[int, int]],
                    glm_offsets: List[Tuple[int, int]]) -> List[List[int]]:
    mapping = [[] for _ in wp_offsets]; p = 0
    for i,(ws,we) in enumerate(wp_offsets):
        while p < len(glm_offsets) and glm_offsets[p][1] <= ws: p += 1
        q = p
        while q < len(glm_offsets) and glm_offsets[q][0] < we:
            mapping[i].append(q); q += 1
    return mapping

class TextFeatureExtractor:
    def __init__(self):
        self.wp_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.glm_tok: AutoTokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        self.fast_offset = True
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.glm = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            quantization_config=quantization_config,
            output_hidden_states=True,
            device_map="auto"
        )
        self.proj_down = nn.Linear(4096, 768, bias=False).to(device).half()
        self.proj_up = nn.Linear(768, 4096, bias=False).to(device).half()
        self.norm = nn.LayerNorm(768).to(device).half()

        for m in (self.proj_down, self.proj_up):
            nn.init.xavier_uniform_(m.weight)

    def _manual_offsets(self, text: str, toks: List[str]):
        norm = text.lower(); p, off = 0, []
        for t in toks:
            tc = t.lstrip(" "); tc = tc if tc else " "
            j = norm.find(tc, p)
            j = j if j != -1 else p
            off.append((j, j + len(tc))); p = j + len(tc)
        return off

    @torch.no_grad()
    def encode(self,
               utterance_text: str,
               knowledge_triples: list | None = None,
               anchor_entities: list | None = None):
        know_parts = []
        if anchor_entities:
            know_parts.append(" [ENT] ".join(map(str, anchor_entities)))
        if knowledge_triples:
            know_parts += [f"{h} [R] {r} [T] {t}" for h, r, t in knowledge_triples]
        knowledge_text = " ".join(know_parts) if know_parts else None

        wp = self.wp_tok(
            utterance_text,
            text_pair=knowledge_text,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        wp = _to_device(wp, self.glm.device)
        wp_tokens = self.wp_tok.convert_ids_to_tokens(wp["input_ids"][0])
        wp_offsets = wp["offset_mapping"][0].tolist()
        token_types = wp["token_type_ids"][0].tolist()
        sep_idx = [i for i, tok in enumerate(wp_tokens) if tok == "[SEP]"]

        knowledge_blocks: List[Tuple[int, int]] = []
        in_block, start = False, None
        for i, (tt, tok) in enumerate(zip(token_types, wp_tokens)):
            if tt == 1 and tok != "[SEP]":
                if not in_block:
                    start, in_block = i, True
            else:
                if in_block:
                    knowledge_blocks.append((start, i))
                    in_block = False
        if in_block:
            knowledge_blocks.append((start, len(wp_tokens)))

        merged = utterance_text + (" [SEP] " + knowledge_text if knowledge_text else "")
        glm_ids = self.glm_tok.encode(merged, add_special_tokens=False)
        glm_enc = {
            "input_ids": torch.tensor([glm_ids], device=self.glm.device),
            "attention_mask": torch.ones(
                1, len(glm_ids), dtype=torch.long, device=self.glm.device)
        }
        glm_tokens = self.glm_tok.convert_ids_to_tokens(glm_ids)
        glm_offsets = (self.glm_tok(
            merged, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
                       if hasattr(self.glm_tok, "get_offsets_mapping")
                       else self._manual_offsets(merged, glm_tokens))

        hidden4096 = self.glm(**glm_enc).hidden_states[-1][0]
        hid768 = self.norm(self.proj_down(hidden4096))

        map_wp2glm = build_cross_map(wp_offsets, glm_offsets)
        max_idx = hid768.size(0)
        wp_emb = torch.stack([
            (
                hid768[torch.tensor(valid, dtype=torch.long,
                                    device=hid768.device)].mean(0)
                if (valid := [i for i in ids if i < max_idx])
                else torch.zeros(768, device=hid768.device)
            )
            for ids in map_wp2glm
        ])

        meta = {
            "wp_tokens": wp_tokens,
            "glm_tokens": glm_tokens,
            "map_wp2glm": map_wp2glm,
            "sep": sep_idx,
            "knowledge_blocks": knowledge_blocks,
        }
        return wp_emb, meta

    def cleanup(self):
        print("TextFeatureExtractor 메모리 해제 중...")
        if hasattr(self, 'glm'):
            del self.glm
        if hasattr(self, 'proj_down'):
            del self.proj_down
        if hasattr(self, 'proj_up'):
            del self.proj_up
        torch.cuda.empty_cache()
        print("TextFeatureExtractor 메모리 해제 완료")

# GraphBuilder 클래스
class GraphBuilder:
    """단일 발화 그래프 + 여러 발화 병합 유틸 (***양방향 엣지 버전***)"""

    # [수정됨] 생성자에서 time_iso 제거
    def __init__(self, *, merge_anchor: bool = False):
        self.merge_anchor = merge_anchor
        self.text_enc = TextFeatureExtractor()
        self.ekm = ExternalFinancialKnowledgeModel()

    def __del__(self):
        if hasattr(self, 'text_enc'):
            self.text_enc.cleanup()
            del self.text_enc
        print("GraphBuilder 메모리 해제 완료")
        log_vram("del", device)

    def _triple_to_string(self, subj: str, pid: str, obj: str) -> str:
        if not hasattr(self, "_pid_cache"):
            self._pid_cache: Dict[str, str] = {}
        if pid not in self._pid_cache:
            row = self.ekm.wiki.execute(
                "SELECT label FROM property_labels WHERE pid=? LIMIT 1", (pid,)
            ).fetchone()
            self._pid_cache[pid] = row[0] if row else pid
        return f"{subj} [REL] {self._pid_cache[pid].replace(' ', '_')} [REL] {obj}"

    # [수정됨] build 메소드가 time_iso를 인자로 받도록 변경
    def build(
        self,
        utterance_text: str,
        time_iso: str,
        video_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
    ) -> Data:
        """발화(텍스트+모달) → PyG Data (모든 엣지 양방향)"""
        if video_emb is not None and video_emb.is_cuda:
            video_emb = video_emb.cpu()
        if audio_emb is not None and audio_emb.is_cuda:
            audio_emb = audio_emb.cpu()

        # 1) 외부 지식 트리플 수집
        # [수정됨] self.time_iso 대신 인자로 받은 time_iso 사용
        ent_ids, ek_sub = self.ekm.acquire_related_external_knowledge(
            utterance_text, time_iso
        )
        triples: List[Tuple[str, str, str]] = []
        if ek_sub.num_edges:
            for s, d, (prop,) in zip(*ek_sub.edge_index, ek_sub.edge_attr):
                triples.append(
                    (
                        self.ekm.int2qid[s.item()],
                        f"P{prop.item()}",
                        self.ekm.int2qid[d.item()],
                    )
                )

        # 2) 텍스트 인코딩
        wp_emb, meta = self.text_enc.encode(
            utterance_text,
            knowledge_triples=[self._triple_to_string(*t) for t in triples],
            anchor_entities=[self.ekm.int2qid[i] for i in ent_ids],
        )
        if wp_emb.is_cuda:
            wp_emb = wp_emb.cpu()
        hs = wp_emb
        D = hs.size(1)
        sep0 = meta["sep"][0] if meta["sep"] else -1

        node_feats: List[torch.Tensor] = []
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []

        # --- utterance text 토큰 노드 ---
        text_token_map: Dict[int, int] = {
            idx: len(node_feats) for idx in range(1, sep0)
        }
        node_feats.extend([hs[i] for i in range(1, sep0)])
        text_nodes = list(text_token_map.values())
        for i in range(len(text_nodes) - 1):
            _add_bidir(
                text_nodes[i], text_nodes[i + 1], EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type
            )

        # --- knowledge 토큰 노드 ---
        knowledge_token_map: Dict[int, int] = {}
        for s, e in meta["knowledge_blocks"]:
            prev = None
            for idx in range(s, e):
                g = len(node_feats)
                knowledge_token_map[idx] = g
                node_feats.append(hs[idx])
                if prev is not None:
                    _add_bidir(prev, g, EDGE_TYPE["t_t"], edge_src, edge_dst, edge_type)
                prev = g

        # 비디오 노드
        if video_emb is not None and video_emb.numel():
            v_idx = len(node_feats)
            node_feats.append(video_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, v_idx, EDGE_TYPE["t_v"], edge_src, edge_dst, edge_type)
        else:
            v_idx = -1

        # 오디오 노드
        if audio_emb is not None and audio_emb.numel():
            a_idx = len(node_feats)
            node_feats.append(audio_emb.squeeze(0))
            for t in text_nodes:
                _add_bidir(t, a_idx, EDGE_TYPE["t_a"], edge_src, edge_dst, edge_type)
        else:
            a_idx = -1

        x = torch.stack(node_feats) if node_feats else torch.empty(0, D)
        edge_index = (
            torch.tensor([edge_src, edge_dst], dtype=torch.long)
            if edge_src
            else torch.empty(2, 0, dtype=torch.long)
        )
        edge_type_t = (
            torch.tensor(edge_type, dtype=torch.long)
            if edge_type
            else torch.empty(0, dtype=torch.long)
        )

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type_t)
        data.node_meta = {
            "text_nodes": len(text_nodes),
            "knowledge_nodes": len(knowledge_token_map),
            "video_nodes": 1 if v_idx != -1 else 0,
            "audio_nodes": 1 if a_idx != -1 else 0,
            "triples": len(triples),
        }
        data.utt_meta = {
            "first_text_node": text_nodes[0] if text_nodes else -1,
            "last_text_node": text_nodes[-1] if text_nodes else -1,
        }
        return data

# models/graph_utils.py의 merge_graph 메소드
def merge_graph(prev_graph: Data | None, current_graph: Data) -> Data:
    if prev_graph is None or prev_graph.x.numel() == 0:
        return current_graph

    # 노드 피처 병합
    x_merged = torch.cat([prev_graph.x, current_graph.x], dim=0)

    # 엣지 인덱스 및 타입 병합 (오프셋 적용)
    num_prev_nodes = prev_graph.x.size(0)
    edge_index_current_offset = current_graph.edge_index + num_prev_nodes
    edge_index_merged = torch.cat([prev_graph.edge_index, edge_index_current_offset], dim=1)

    edge_type_merged = torch.cat([prev_graph.edge_type, current_graph.edge_type])

    # utterance 연결 엣지 추가 (양방향)
    if prev_graph.utt_meta["last_text_node"] != -1 and current_graph.utt_meta["first_text_node"] != -1:
        prev_last_node = prev_graph.utt_meta["last_text_node"]
        curr_first_node = current_graph.utt_meta["first_text_node"] + num_prev_nodes
        
        utt_edge_index = torch.tensor([[prev_last_node, curr_first_node],
                                       [curr_first_node, prev_last_node]], dtype=torch.long)
        utt_edge_type = torch.tensor([EDGE_TYPE["utt"], EDGE_TYPE["utt"]], dtype=torch.long)

        edge_index_merged = torch.cat([edge_index_merged, utt_edge_index], dim=1)
        edge_type_merged = torch.cat([edge_type_merged, utt_edge_type])

    # 메타 정보 병합
    node_meta_merged = {k: prev_graph.node_meta.get(k, 0) + current_graph.node_meta.get(k, 0)
                        for k in set(prev_graph.node_meta.keys()) | set(current_graph.node_meta.keys())}
    
    utt_meta_merged = {
        "first_text_node": prev_graph.utt_meta["first_text_node"],
        "last_text_node": current_graph.utt_meta["last_text_node"] + num_prev_nodes
    }

    merged_graph = Data(x=x_merged, edge_index=edge_index_merged, edge_type=edge_type_merged)
    merged_graph.node_meta = node_meta_merged
    merged_graph.utt_meta = utt_meta_merged
    return merged_graph

def build_and_cache_graphs():
    """
    ready_videos.csv에 있는 video_id에 대해 그래프를 구축하고 캐시로 저장합니다.
    """
    # 파일 경로 설정
    ready_videos_path = "data/ready_videos.csv"
    speech_db_path = "data/speech_segments.db"
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # 1. ready_videos.csv에서 video_id 목록 불러오기
    try:
        ready_videos_df = pd.read_csv(ready_videos_path)
        video_ids = ready_videos_df["video_id"].tolist()
        print(f"총 {len(video_ids)}개의 video_id를 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: {ready_videos_path} 파일을 찾을 수 없습니다.")
        return

    # 2. speech_segments.db 연결
    try:
        conn = sqlite3.connect(speech_db_path)
        conn.row_factory = sqlite3.Row
        print(f"{speech_db_path}에 연결되었습니다.")
    except sqlite3.Error as e:
        print(f"오류: {speech_db_path} 연결 실패 - {e}")
        return

    # [추가됨] GraphBuilder 인스턴스를 루프 밖에서 한 번만 생성
    print("GraphBuilder 인스턴스를 생성합니다. 모델 로딩으로 시간이 소요될 수 있습니다...")
    try:
        graph_builder = GraphBuilder()
    except Exception as e:
        print(f"오류: GraphBuilder 인스턴스 생성 실패 - {e}")
        conn.close()
        return

    # 3. 각 video_id에 대해 그래프 구축 및 캐시 저장
    for video_id in video_ids:
        # 🟢 추가된 로직: 최종 그래프 파일이 이미 존재하는지 확인
        final_graph_path = cache_dir / _hash(f"video_graph::{video_id}")
        if final_graph_path.exists():
            print(f"\n[Video ID: {video_id}] 최종 그래프가 이미 존재합니다. 건너뜁니다.")
            continue
            
        print(f"\n[Video ID: {video_id}] 그래프 구축 시작...")
        
        # 비디오 업로드 시간 조회
        try:
            video_meta = conn.execute(
                "SELECT published_date FROM video_metadata WHERE video_id = ?",
                (video_id,)
            ).fetchone()
            if not video_meta:
                print(f"경고: video_id {video_id}에 대한 메타데이터를 찾을 수 없습니다. 건너뜁니다.")
                continue
            upload_time = video_meta["published_date"]
        except sqlite3.Error as e:
            print(f"오류: video_id {video_id}의 업로드 시간 조회 실패 - {e}")
            continue

        # [수정됨] 루프 내에서 GraphBuilder를 생성하는 대신, 이미 생성된 객체 사용
        # graph_builder = GraphBuilder(time_iso=upload_time) # 이 라인 제거

        video_graph = None

        # 비디오에 대한 발화 불러오기
        speech_segments = conn.execute(
            "SELECT segment_id, script FROM speech_segments WHERE video_id = ? ORDER BY start_time",
            (video_id,)
        ).fetchall()
        
        if not speech_segments:
            print(f"경고: video_id {video_id}에 대한 발화가 없습니다. 그래프를 생성하지 않습니다.")
            continue
        
        for segment in speech_segments:
            seg_id = segment["segment_id"]
            utterance_text = segment["script"]
            
            # [추가됨] 빈 발화 텍스트 건너뛰기
            if not utterance_text or not utterance_text.strip():
                print(f" - 경고: 발화 '{seg_id}'의 텍스트가 비어있어 건너뜁니다.")
                continue
                
            print(f" - 발화 '{seg_id}' 처리 중...")

            # 캐시 파일 경로 설정
            video_key = f"vid::{seg_id}"
            audio_key = f"aud::{seg_id}"
            video_cache_path = cache_dir / _hash(video_key)
            audio_cache_path = cache_dir / _hash(audio_key)

            # 임베딩 텐서 로드
            video_emb, audio_emb = None, None
            try:
                if video_cache_path.exists():
                    video_emb = torch.load(video_cache_path)
                if audio_cache_path.exists():
                    audio_emb = torch.load(audio_cache_path)
            except Exception as e:
                print(f"오류: 발화 '{seg_id}'의 캐시 파일 로드 실패 - {e}")
                continue

            # 단일 발화 그래프 구축
            try:
                # [수정됨] build 메소드에 time_iso 인자 전달
                current_graph = graph_builder.build(
                    utterance_text=utterance_text,
                    time_iso=upload_time, 
                    video_emb=video_emb,
                    audio_emb=audio_emb
                )
            except Exception as e:
                print(f"오류: 발화 '{seg_id}'에 대한 그래프 구축 실패 - {e}")
                continue

            # 이전 그래프와 병합
            video_graph = merge_graph(video_graph, current_graph)

        # 전체 비디오 그래프 캐시 저장
        if video_graph:
            # 🟢 추가된 로직: 이전에 존재하지 않았던 경우에만 저장
            torch.save(video_graph, final_graph_path)
            print(f"\n성공: video_id {video_id}에 대한 최종 그래프가 {final_graph_path}에 저장되었습니다.")
        
        # [수정됨] 루프 내에서 del graph_builder 제거

    # [추가됨] 모든 작업이 끝난 후 GraphBuilder 인스턴스 명시적으로 삭제
    print("\n모든 비디오 처리가 완료되었습니다. GraphBuilder 리소스를 해제합니다.")
    del graph_builder
    torch.cuda.empty_cache()

    conn.close()
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    build_and_cache_graphs()