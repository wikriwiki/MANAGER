# from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset
import torch
import pandas as pd
from feature_extractor import TextFeatureExtractor

text_model = TextFeatureExtractor()

class DataLoader_Graph(Dataset):
    def __init__(self, annotation_path):
        super(DataLoader_Graph, self).__init__()
        self.df = pd.read_csv(annotation_path)
        self.emb_file = self.df['emb_file']
        self.graph_file = self.df['graph_file']
        # self.question = self.df['matching_questions']
        self.label = self.df['threshold_label']
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        emb = torch.load(self.emb_file[idx])
        graph = torch.load(self.graph_file[idx], weights_only=False).edge_index
        # question = self.question[idx]
        answer = self.label[idx]
        
        return emb, graph, answer
    
class DataLoader(Dataset):
    def __init__(self, annotation_path):
        super(DataLoader, self).__init__()
        self.df = pd.read_csv(annotation_path) #[title,url,matching_questions,similarity,emb_file,graph_file]
        self.text_emb_file = self.df['text_file']
        self.audio_emb_file = self.df['audio_file']
        self.video_emb_file = self.df['video_file']
        # self.question_emb_file = self.df['question_file']
        self.label = self.df['threshold_label']
        self.y = self.label.values
        self.data = None

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # row = self.df.iloc[idx]
        # title = row['title']
        # questions = row['matching_questions']
        # q_emb = text_model.encode(questions)
        # emb = torch.load(row['emb_file'])#.to(torch.float16)  # Ensure the embedding is a float tensor

        t_emb = torch.load(self.text_emb_file[idx])
        a_emb = torch.load(self.audio_emb_file[idx]).float()
        v_emb = torch.load(self.video_emb_file[idx])
        # q_emb = torch.load(self.question_emb_file[idx])
        # v_emb = torch.load(row['video_file'])
        # t_emb = torch.load(row['text_file'])
        # q_emb = torch.load(row['question_file'])
        # edge_index = torch.load(row['graph_file']).edge_index
        label = torch.tensor([1 if self.label.values[idx] == "Impact" else 0])  #, dtype=torch.float16 float for BCEWithLogitsLoss

        # data = {
        #     "text": t_emb,
        #     "audio": a_emb,
        #     "video": v_emb,
        #     "question": q_emb,
        #     # "edge_index": edge_index,
        #     "label": label
        # }

        return t_emb,a_emb,v_emb,label

    # def __call__(self):
    #     self.data = []
    #     for i in range(self.data_len):
    #         emb_file = self.df_annotation.iloc[i]['emb_file']
    #         graph_file = self.df_annotation.iloc[i]['graph_file']
    #         label = self.df_annotation.iloc[i]['label']

    #         emb = torch.load(emb_file)
    #         graph = torch.load(graph_file)
    #         label = torch.tensor(label, dtype=torch.long)

    #         # Create Data object
    #         data = Data(x=emb, edge_index=graph, y=label)

    #         # Append to the list
    #         self.data.append(data)

    #     return self.data
