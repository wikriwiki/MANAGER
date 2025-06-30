import torch
import pandas as pd
from torch_geometric.data import Data
from feature_extractor import ExternalFinancialKnowledgeModel, TextFeatureExtractor
from tqdm import tqdm

class RelationConstructor:
    def __init__(self, time_idx=100):
        self.time_idx = 100
        self.external_model = ExternalFinancialKnowledgeModel()

        
    def load_data(self):
        """
        Load the data from the given paths and process it.
        """
        # Load utterance raw data
        self.utterance_raw_data = pd.read_csv(self.utterance_raw_path)  #['Text'].tolist() # Utterance Segment List (N_U) [Start Time (s),End Time (s),Text]

        # drop Start time >= End Time
        self.utterance_raw_data = self.utterance_raw_data[self.utterance_raw_data['Start Time (s)'] < self.utterance_raw_data['End Time (s)']]
        self.utterance_raw_data = self.utterance_raw_data['Text'].tolist()
        
        # Load audio data
        self.audio_data = torch.load(self.audio_path) # Audio Embedding File (N_U, 4096)
        
        # Load video data
        self.video_data = torch.load(self.video_path) # Video Embedding File (N_U, 4096)

        self.utt_len = len(self.utterance_raw_data)

        

    def token_token_edges(self):
        """
        Create token-token edges based on the utterances data.
        """
        # Create a Data object to store the graph
        data = Data()

        text_emb_model = TextFeatureExtractor()

        emb_list = []
        edges = []

        for utt_idx,text in tqdm(enumerate(self.utterance_raw_data),desc="Token to token precessing"):
            result = text_emb_model.encode(text)
            last_hidden = result.squeeze()
            emb_list.append(torch.mean(last_hidden,dim=0))

        # Create a list of edges (source, target)
            if utt_idx < self.utt_len - 1:
                # Connect current utterance to the next utterance
                edges.append((utt_idx, utt_idx + 1))

        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Store the edge index in the Data object
        data.edge_index = edge_index

        emb_list = torch.stack(emb_list,dim=0)

        return emb_list, data
    
    def token_knowledge_edges(self, utt_graph:Data):

        text_emb_model = TextFeatureExtractor()

        graph_offset = len(utt_graph.edge_index[1])
        emb_list = []

        for utt_idx,text in tqdm(enumerate(self.utterance_raw_data),desc="Token to knowledge precessing"):
            # utt_idx = utterance node index in data
            # Load external financial knowledge model data
            entities_id, external_data = self.external_model.acquire_related_external_knowledge(text, self.time_idx)

            external_src_id = external_data.edge_index[0]
            external_relation_id = external_data.edge_attr
            external_dst_id = external_data.edge_index[1]

            new_edges = []

            for i in range(len(external_src_id)):
                source = self.external_model.id_to_entity(external_src_id[i].item())
                relation = self.external_model.id_to_relation(external_relation_id[i].item())
                destination = self.external_model.id_to_entity(external_dst_id[i].item())

                text = " ".join([source,relation,destination])

                result = text_emb_model.encode(text)
                last_hidden = result.squeeze()
                emb_list.append(torch.mean(last_hidden,dim=0))

                # new node(graph_offset) -> cur_utt_node(utt_idx)
                new_node_idx = graph_offset + i
                new_edges.append([new_node_idx, utt_idx])  # external node → utterance node

            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            utt_graph.edge_index = torch.cat([utt_graph.edge_index, new_edges_tensor], dim=1)

            graph_offset += len(external_src_id)

        emb_list = torch.stack(emb_list,dim=0)


        return emb_list, utt_graph
    
    def token_video_edges(self, graph:Data):

        graph_offset = len(graph.edge_index[1])
        new_edges = []

        for i in range(self.utt_len):

            new_edges.append([graph_offset + i, i])

        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        graph.edge_index = torch.cat([graph.edge_index, new_edges_tensor], dim=1)

        return self.video_data, graph
    
    def token_audio_edges(self, graph:Data):
        graph_offset = len(graph.edge_index[1])
        new_edges = []

        for i in range(self.utt_len):
            new_edges.append([graph_offset + i, i])

        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        graph.edge_index = torch.cat([graph.edge_index, new_edges_tensor], dim=1)

        return self.audio_data, graph


    def get_relation_graph(self, utterance_raw_path, utterance_path, audio_path, video_path,):
        self.utterance_raw_path = utterance_raw_path
        self.utterance_path = utterance_path
        self.audio_path = audio_path
        self.video_path = video_path

        self.load_data()

        emb_list = []
        utt_emb, graph_data = self.token_token_edges()
        knowledge_emb, graph_data = self.token_knowledge_edges(graph_data)
        video_emb, graph_data = self.token_video_edges(graph_data)
        audio_emb, graph_data = self.token_audio_edges(graph_data)

        # print(utt_emb.shape)
        # print(knowledge_emb.shape)
        # print(video_emb.shape)
        # print(audio_emb.shape)

        emb_list.extend([t.cpu() for t in utt_emb])
        emb_list.extend([t.cpu() for t in knowledge_emb])
        emb_list.extend([t.cpu() for t in video_emb])
        emb_list.extend([t.cpu() for t in audio_emb])

        return torch.stack(emb_list,dim=0), graph_data

    
if __name__ == "__main__":
    RC = RelationConstructor()

    # emb, graph = RC.get_relation_graph("output/utterance.csv","output/utterance.pt","output/audio_features.pt","output/video_features.pt")

    # torch.save(emb,"output/emb.pt")
    # torch.save(graph,"output/graph.pt")

    # print(emb.shape)

    # print(graph)

    graph = torch.load("output/graph.pt")

    RC.external_model.draw_graph(graph)