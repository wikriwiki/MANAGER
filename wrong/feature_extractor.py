import torch
import pandas as pd
from torch_geometric.data import Data
from transformers import BeitImageProcessor, BeitModel, AutoProcessor, HubertModel, pipeline, AutoFeatureExtractor, AutoTokenizer, AutoModel
from PIL import Image
import re
import librosa
import moviepy as mp
import io
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_memory_usage(model):
    """ 모델이 차지하는 GPU 메모리 크기 출력 (GB 단위) """
    model_parameters = sum(p.numel() for p in model.parameters())  # 총 파라미터 개수
    model_memory_bytes = model_parameters * 4  # float32(4 bytes) 기준
    model_memory_gb = model_memory_bytes / (1024 ** 3)  # GB 변환

    print(f"Model Parameters: {model_parameters:,}")  # 1000 단위 콤마 추가
    print(f"Model GPU Memory Usage: {model_memory_gb:.6f} GB")

def print_gpu_memory_usage():
    """ 현재 할당된 모델의 GPU 메모리 사용량 출력 """
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위 변환
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB 단위 변환

    print(f" 현재 할당된 GPU 메모리: {allocated:.4f} GB") 
    print(f" 현재 예약된 GPU 메모리: {reserved:.4f} GB")

class ExternalFinancialKnowledgeModel:
    def __init__(self):
        self.name = "ExternalFinancialKnowledgeModel"
        self.model = None

        file_path = "FinDKG_dataset/FinDKG-full"

        self.df_findkg = pd.read_csv(f"{file_path}/train.txt",sep="\t",header=None)
        self.df_findkg.columns = ['subject','relation','object','time_id','ignored']

        df_tmp = pd.read_csv(f"{file_path}/valid.txt",sep="\t",header=None)
        df_tmp.columns = ['subject','relation','object','time_id','ignored']

        self.df_findkg = pd.concat([self.df_findkg,df_tmp])

        df_tmp = pd.read_csv(f"{file_path}/test.txt",sep="\t",header=None)
        df_tmp.columns = ['subject','relation','object','time_id','ignored']

        self.df_findkg = pd.concat([self.df_findkg,df_tmp])

        self.df_entity_id = pd.read_csv(f"{file_path}/entity2id.txt",sep="\t",header=None)
        self.df_entity_id.columns = ['entity','id','id2','time_id']

        self.df_relation_id = pd.read_csv(f"{file_path}/relation2id.txt",sep="\t",header=None)
        self.df_relation_id.columns = ['relation','id']

        self.df_time_id = pd.read_csv(f"{file_path}/time2id.txt",sep=",",header=[0])
        self.df_time_id.columns = ['id','time']

        self.target_entities = self.df_entity_id["entity"].tolist()
    

    def get_graph_before_time(self,time):
        # 특정 time 값 이하인 데이터 필터링
        df_filtered = self.df_findkg[self.df_findkg["time_id"] <= time]

        # 엣지 리스트 생성
        edge_index = torch.tensor([df_filtered["subject"].tolist(), df_filtered["object"].tolist()], dtype=torch.long)

        # 엣지 속성 (관계)
        edge_attr = torch.tensor(df_filtered["relation"].tolist(), dtype=torch.long).view(-1, 1)

        # PyG 데이터 객체 생성
        graph_data = Data(edge_index=edge_index, edge_attr=edge_attr)

        return graph_data

    def identify_entities(self, text):
        # 특수문자 제거 (단어와 공백 제외)
        text = re.sub(r"[^\w\s]", "", text.lower())

        # entity 정규식 패턴 생성 (\s+ 허용)
        patterns = [re.sub(r'\s+', r'\\s+', re.escape(ent.lower())) for ent in self.target_entities]
        regex_pattern = r'(' + '|'.join(patterns) + r')'

        # 정규식 매칭
        matches = re.findall(regex_pattern, text)

        # 알파벳 + 공백만 허용
        return [
            entity.strip() for entity in set(matches)
            if re.fullmatch(r"[a-z\s]+", entity.strip())
        ]
    
    def entities_to_id(self,entities) -> list:
        entity_id = []
        for entity in entities:
            # 대소문자 구분 없이 찾기 위해 lower() 사용
            matching_entity = self.df_entity_id[self.df_entity_id["entity"].str.lower() == entity.lower()]
            
            # 일치하는 엔티티가 있을 경우 첫 번째 id 값 가져오기
            if not matching_entity.empty:
                entity_id.append(matching_entity["id"].values[0])

        return entity_id
    
    def id_to_entity(self,id) -> str:
        return self.df_entity_id[self.df_entity_id["id"]==id]["entity"].values[0]
    
    def id_to_relation(self,id) -> str:
        return self.df_relation_id[self.df_relation_id["id"]==id]["relation"].values[0]
    
    def acquire_related_external_knowledge(self,text,time) -> Tuple[List[int], Data]:
        # 텍스트 내에서 엔티티 추출
        entities = self.identify_entities(text)

        # print(f"Identified entities: {entities}")

        # 엔티티 ID 획득
        entities_id = self.entities_to_id(entities)

        # print(f"Identified entities ID: {entities_id}")

        # 특정 시점 이전의 그래프 데이터 획득
        graph_data = self.get_graph_before_time(time)

        # 1-hop 서브그래프 추출
        src_nodes = graph_data.edge_index[0]
        dst_nodes = graph_data.edge_index[1]
        mask = torch.isin(src_nodes, torch.tensor(entities_id)) | torch.isin(dst_nodes, torch.tensor(entities_id))
        sub_edge_index = graph_data.edge_index[:, mask]
        sub_edge_attr = graph_data.edge_attr[mask]
        
        subgraph_data = Data(edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return entities_id, subgraph_data
    
    def draw_graph(self,graph_data):
        G = nx.DiGraph()
        
        edge_index = graph_data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        
        G.add_edges_from(edges)
        
        plt.figure(figsize=(8, 6))
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Graph Visualization")
        plt.show()

class TextFeatureExtractor:
    def __init__(self):
        self.name = "BERTTextFeatureExtractor"

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.to(device)

    def encode(self, text:str):
        # 토큰화 및 텐서 변환
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # 모델을 통해 임베딩 추출
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 마지막 히든 스테이트 추출
        last_hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)


        # 문장 임베딩으로 [CLS] 토큰의 벡터 사용
        sentence_embedding = last_hidden_states[:, 0, :]  # shape: [batch_size, 768]
        # sentence_embedding = last_hidden_states.mean(dim=1)  # shape: [batch_size, 768]

        return last_hidden_states
        

class VideoFeatureExtractor:
    def __init__(self, model="microsoft/beit-base-patch16-224"):
        self.name = "BEiT3VideoFeatureExtractor"
        
        # BEiT-3 모델과 이미지 프로세서 로드
        self.processor = BeitImageProcessor.from_pretrained(model)
        self.model = BeitModel.from_pretrained(model)
        self.model.to(device)

    def encode(self, image: Image.Image):
        """
        이미지에서 특징을 추출하는 함수입니다.
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            torch.Tensor: 추출된 특징 벡터
        """
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt").to(device)

        # 모델을 통해 특징 추출
        with torch.no_grad():
            outputs = self.model(**inputs) # [batch_size, sequence_length, hidden_size]
        
        # 최종 특징 벡터 (CLS 토큰 사용)
        features = outputs.last_hidden_state[:, 0, :]
        return features
    
    def aggregate_encoding_feature(self, frames):
        """
        주어진 프레임 리스트를 인코딩하고 평균 풀링을 통해 특징을 집계합니다.

        Args:
            frames (list[PIL.Image]): PIL Image 객체로 구성된 리스트

        Returns:
            torch.Tensor: 모든 프레임의 특징을 평균 풀링한 결과
        """
        encoded_features = [self.encode(frame) for frame in frames]
        return torch.stack(encoded_features)

    def extract_from_video(self, video_frames_folder):
        """
        비디오 프레임 폴더에서 프레임을 읽고 특징을 추출합니다.

        Args:
            video_frames_folder (str): 프레임 이미지가 저장된 폴더 경로

        Returns:
            torch.Tensor: 집계된 특징 벡터
        """
        frame_files = sorted([f for f in os.listdir(video_frames_folder) if f.endswith(".jpg") or f.endswith(".png")])
        frames = [Image.open(os.path.join(video_frames_folder, file)).convert("RGB") for file in frame_files]
        
        return self.aggregate_encoding_feature(frames)

    def save_features(self, features, output_path):
        """
        추출된 특징 벡터를 저장합니다.

        Args:
            features (torch.Tensor): 저장할 특징 벡터
            output_path (str): 저장할 파일 경로
        """
        torch.save(features, output_path)
        print(f"Features saved to {output_path}")

class AudioFeatureExtractor:
    def __init__(self, model="facebook/hubert-base-ls960",audio_folder="output/split_audio"):
        self.name = "AudioFeatureExtractor"
        self.audio_folder = audio_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HuBERT 모델과 프로세서 로드
        self.processor = AutoFeatureExtractor.from_pretrained(model)
        self.model = HubertModel.from_pretrained(model).to(self.device)
        self.model.eval()
        self.model.half()
        # self.model = pipeline("feature-extraction", model="facebook/hubert-base-ls960",torch_dtype=torch.float16,device="cuda")

    def encode(self, data, sampling_rate=16000):
        """
        오디오 데이터를 특징 벡터로 변환하는 함수입니다.
        
        Args:
            data (Tensor): 오디오 데이터를 나타내는 Tensor
            sampling_rate (int): 샘플링 레이트 (기본값: 16000Hz)
        
        Returns:
            torch.Tensor: 추출된 특징 벡터
        """
        # 오디오 데이터를 HuBERT 프로세서에 맞게 전처리
        inputs = self.processor(data, return_tensors="pt", sampling_rate=sampling_rate).to(self.device)
         
        # 모델을 통해 특징 추출
        with torch.no_grad():
            outputs = self.model(**inputs.to(torch.float16))
        
        # 마지막 hidden state에서 특징 벡터 추출 [CLS]
        features = outputs.last_hidden_state
        return features[:,0,:]

    def extract_from_audio_folder(self, audio_path):
        """
        분할된 오디오 파일을 불러와 특징을 추출하고 저장합니다.
        """
        audio_files = sorted([f for f in os.listdir(audio_path) if f.endswith(".wav")])

        features = []
        
        for audio_file in audio_files:
            file_path = os.path.join(audio_path, audio_file)
            # print(f"Processing: {file_path}")
            
            # 오디오 파일 로드
            audio_data, sr = librosa.load(file_path, sr=16000)
            audio_tensor = torch.tensor(audio_data).to(self.device)  # (1, Samples)
            
            # 특징 벡터 추출
            features.append(self.encode(audio_tensor, sr))

        return torch.stack(features, dim=0)
    
# class MultimodalFeatureExtractor:
#     def __init__(self):
#         self.model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")

#     def extract_features(self, text, video_frames_folder, audio_folder):
#         text_features = self.text_extractor.encode(text)
#         video_features = self.video_extractor.extract_from_video(video_frames_folder)
#         audio_features = self.audio_extractor.extract_from_audio_folder(audio_folder)

#         return text_features, video_features, audio_features
    
if __name__ == "__main__":
    model = ExternalFinancialKnowledgeModel()

    # text = '''
    #     Donald Trump, the former president of the United States, has often expressed strong opinions about China,
    #     particularly regarding trade policies and global economic competition.'''

    # sub_graph = model.acquire_related_external_knowledge(text,1)

    # print(sub_graph)

    sub_graph = torch.load("data/graph_0505.pt")

    model.draw_graph(sub_graph)

    # video_feature_extractor = VideoFeatureExtractor()
    # audio_feature_extractor = AudioFeatureExtractor()
    # frame_path = "output/extracted_frames"

    # audio_features = audio_feature_extractor.extract_from_audio_folder().to("cpu").squeeze() # X_A [ N_U, 768 ]

    # # Save audio features
    # audio_features_path = "output/audio_features.pt"
    # print(f"Audio features shape: {audio_features.shape}")
    # torch.save(audio_features, audio_features_path)

    # video_features = video_feature_extractor.extract_from_video(frame_path).to("cpu").squeeze()  # X_V [ N_U, 768 ]

    # # Save video features
    # video_features_path = "output/video_features.pt"
    # print(f"Video features shape: {video_features.shape}")
    # torch.save(video_features, video_features_path)
    
