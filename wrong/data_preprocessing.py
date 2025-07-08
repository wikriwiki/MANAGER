import whisper
import librosa
import soundfile as sf
import cv2
import os
import csv
import pandas as pd
import subprocess
from tqdm import tqdm
from feature_extractor import ExternalFinancialKnowledgeModel, TextFeatureExtractor, AudioFeatureExtractor, VideoFeatureExtractor
from semantic_relation_constructor import RelationConstructor
from moviepy import VideoFileClip
import torch
import shutil
import json

def download_video(url, output_path, i):
    """
    주어진 url에서 비디오를 다운로드하여 출력 경로에 저장합니다.

    매개변수:
    url (str): 다운로드할 비디오의 URL입니다.
    output_path (str): 비디오를 저장할 경로입니다.

    반환값:
    dict: 비디오의 메타데이터가 포함된 사전입니다.
    """
    # 경로 생성
    os.makedirs(output_path, exist_ok=True)

    # 비디오 재생 시간 확인
    command = f'yt-dlp --get-duration --cookies-from-browser firefox {url}'
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except:
        return 3601
    duration = result.stdout.strip()

    # 재생 시간이 1시간 이상인 경우 함수 종료
    time_parts = duration.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 2:
        hours = 0
        minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(time_parts[0])
    else:
        raise ValueError("Unexpected duration format")
    total_seconds = hours * 3600 + minutes * 60 + seconds

    if total_seconds > 3600:
        return total_seconds
    
    else:
        # yt-dlp 명령 실행
        try:
            command = f'yt-dlp --force-overwrites -f "bestvideo[height=240]+bestaudio" --cookies-from-browser firefox --user-agent "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0" -o "{output_path}/video_{i}" {url}'
            subprocess.run(command, shell=True, check=True)

            return total_seconds
        except:
            return 3601
        
def convert_webm_to_mp4(video_path,converted_video_path):
        """OpenCV를 사용하여 WebM을 MP4로 변환"""
        # print("WebM을 MP4로 변환 중...")
        clip = VideoFileClip(video_path)
        clip.write_videofile(converted_video_path, codec="libx264", audio_codec="aac")

class NewsAudioVideoProcessor:
    def __init__(self, output_folder="output"):
        self.video_path = None
        self.audio_path = None
        self.output_folder = output_folder
        self.audio_output_folder = os.path.join(output_folder, "split_audio")
        self.frame_output_folder = os.path.join(output_folder, "extracted_frames")
        self.csv_path = os.path.join(self.audio_output_folder, "segments.csv")
        self.converted_video_path = "converted_video.mp4"

        # 폴더 생성
        os.makedirs(self.audio_output_folder, exist_ok=True)
        os.makedirs(self.frame_output_folder, exist_ok=True)

    def convert_webm_to_mp4(self):
        """OpenCV를 사용하여 WebM을 MP4로 변환"""
        # print("WebM을 MP4로 변환 중...")
        clip = VideoFileClip(self.video_path)
        clip.write_videofile(self.converted_video_path, codec="libx264", audio_codec="aac")

        self.video_path = self.converted_video_path
    
    def extract_audio_from_video(self):
        """
        비디오에서 오디오를 추출하여 WAV 파일로 저장합니다.
        입력: self.converted_video_path (str) → mp4 비디오 경로 (확장자 제외)
            self.audio_path (str) → 저장할 오디오 경로 (확장자 제외)
        결과: self.audio_path.wav 로 오디오 파일 저장
        """
        input_video = f"{self.converted_video_path}"
        output_audio = f"{self.audio_path}"

        command = [
            "ffmpeg",
            "-i", input_video,
            "-y",                # 기존 파일 덮어쓰기
            "-vn",                # 비디오 제외
            "-acodec", "pcm_s16le",  # WAV 포맷 (16-bit)
            "-ar", "16000",       # 샘플링 레이트 16kHz
            "-ac", "1",           # 모노 오디오
            output_audio
        ]

        subprocess.run(command, check=True)

    def process_audio(self, model_size="base"):
        """ Whisper 모델을 사용하여 오디오를 발화 단위로 분할하고 저장 """
        # print("오디오 처리 시작...")

        # Whisper 모델 로드
        model = whisper.load_model(model_size)

        # 오디오 로드 및 16kHz 변환
        y, sr = librosa.load(self.audio_path, sr=16000)

        # Whisper를 사용하여 발화 단위 분할
        result = model.transcribe(self.audio_path, word_timestamps=True)

        # 발화 텍스트를 utterance.csv로 저장
        utterance_csv_path = os.path.join(self.audio_output_folder, "utterance.csv")
        with open(utterance_csv_path, mode="w", newline="") as utterance_csv_file:
            utterance_writer = csv.writer(utterance_csv_file)
            utterance_writer.writerow(["Segment Index", "Start Time (s)", "End Time (s)", "Text"])  # 헤더 추가

            for idx, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                if (end_time - start_time) < 1:
                    continue

                # CSV 파일에 저장
                utterance_writer.writerow([idx + 1, start_time, end_time, text])

        # print(f"발화 텍스트 저장 완료: {utterance_csv_path}")

        # CSV 파일 생성
        with open(self.csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Filename", "Start Time (s)", "End Time (s)"])  # 헤더 추가

            # 발화 단위로 오디오 분할 및 저장
            for idx, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]

                if (end_time - start_time) < 1:
                    continue

                # 샘플 단위로 변환 (16kHz 기준)
                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)

                # 오디오 추출
                utterance_audio = y[start_sample:end_sample]

                # 파일 이름 설정 (audio_001.wav, audio_002.wav, ...)
                output_file = os.path.join(self.audio_output_folder, f"audio_{idx+1:03d}.wav")

                # 오디오 저장
                sf.write(output_file, utterance_audio, 16000)

                # CSV 파일에 저장
                csv_writer.writerow([f"audio_{idx+1:03d}.wav", start_time, end_time])

                # print(f"Saved: {output_file} [{start_time:.2f}s - {end_time:.2f}s]")

        # print(f"오디오 분할 완료. CSV 파일 저장됨: {self.csv_path}")

    def extract_frames(self):
        """ 비디오에서 segments.csv에 기록된 시작 시간에 해당하는 프레임을 OpenCV를 사용하여 추출하고 저장 """
        # print("비디오 프레임 추출 시작...")

        # OpenCV를 사용하여 비디오 파일 열기
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("비디오 파일을 열 수 없습니다.")
            return

        # 비디오의 FPS 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"비디오 FPS: {fps}")

        # CSV 파일 읽기
        with open(self.csv_path, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 헤더 건너뛰기

            for idx, row in enumerate(csv_reader):
                filename, start_time, _ = row
                start_time = float(start_time)

                # 해당 시간의 프레임 인덱스 계산
                frame_index = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                # 프레임 읽기
                ret, frame = cap.read()
                if ret:
                    # 저장할 파일 이름 설정 (frame_001.jpg, frame_002.jpg, ...)
                    frame_file = os.path.join(self.frame_output_folder, f"frame_{idx+1:03d}.jpg")

                    # 프레임 저장
                    cv2.imwrite(frame_file, frame)
                    # print(f"Saved: {frame_file} at {start_time:.2f}s")
                else:
                    print(f"프레임 추출 실패: {start_time:.2f}s")

        # 비디오 객체 해제
        cap.release()
        # print("모든 프레임 추출 완료.")

    def init_directory(self):
        """ 결과 저장을 위한 디렉토리 초기화: 기존 폴더 삭제 후 재생성 """
        for folder in [self.audio_output_folder, self.frame_output_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)  # 폴더 및 내부 파일 전체 삭제
            os.makedirs(folder)

    def process(self, video_path:str, audio_path:str):
        """ 전체 프로세스 실행: WebM 변환 -> 오디오 분할 -> 프레임 추출 """
        self.video_path = video_path
        self.audio_path = audio_path
        self.init_directory()
        self.convert_webm_to_mp4()
        self.extract_audio_from_video()
        self.process_audio()
        self.extract_frames()

# class MultiModalEmbedding:
#     def __init__(self, text_model:TextFeatureExtractor, audio_model:AudioFeatureExtractor, video_model:VideoFeatureExtractor):
#         self.text_model = text_model
#         self.audio_model = audio_model
#         self.video_model = video_model

#     def extract_audio_features(self, audio_path):
#         """ 오디오 특징 추출 """
#         audio_features = self.audio_model.extract_from_audio_folder(audio_path)
#         return audio_features

#     def extract_video_features(self, video_path):
#         """ 비디오 특징 추출 """
#         video_features = self.video_model.extract_from_video(video_path)
#         return video_features

#     def extract_text_features(self, text_path):
#         """ 텍스트 특징 추출 """
#         text_features = self.text_model.encode(text)
#         return text_features

#     def processing(self):

def json_to_multimodal_embedding(json_dir, video_dir, output_dir, verbose=False):
        
    text_extractor = TextFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    video_extractor = VideoFeatureExtractor()
    
    json_files = os.listdir(json_dir)
    
    
    
    for json_file in tqdm(json_files):
        base_name = os.path.basename(json_file)
        with open(json_file,"r") as f:
            json_file = json.load(f)
        
        if verbose:
            print(json_file[:3])
            
        video_path = os.path.join(video_dir,f"{base_name}.mp4")
        
        if not os.path.exists(video_path):
            convert_webm_to_mp4(os.path.join(video_dir,f"{base_name}.webm"), video_path)
            
        # Video Processing
        
            
            
    
    
        
    
# 실행 예제
if __name__ == "__main__":
    video_path = "video.webm"
    audio_path = "audio.wav"
    frame_path = "output/extracted_frames"
    audio_features_path = "output/audio_features.pt"
    video_features_path = "output/video_features.pt"

    processor = NewsAudioVideoProcessor()
    audio_model = AudioFeatureExtractor()
    video_model = VideoFeatureExtractor()
    relation_constructor = RelationConstructor()

    df_samples = pd.read_csv("matching_questions_politics_economy_concat.csv")
    df_samples.sort_values(by=['url'], inplace=True)
    df_samples_file = df_samples.copy()

    title_list = []
    url_list = []
    # question_list = []
    emb_file_list = []
    graph_file_list = []

    prev_url = None
    prev_i = None

    for i in tqdm(range(len(df_samples)),desc="Data Processing"):
        url = df_samples.iloc[i]['url']
        title = df_samples.iloc[i]['title']
        # question = df_samples.iloc[i]['matching_questions']

        if prev_url != None and prev_url == url:
            title_list.append(title)
            url_list.append(url)
            # question_list.append(question)
            emb_file_list.append(f"data/embedding/emb_{prev_i+1:04d}.pt")
            graph_file_list.append(f"data/graph/graph_{prev_i+1:04d}.pt")
            continue
        
        if not os.path.exists(f"data/video/video_{i}.webm"):
            download_video(url, "data/video",i)

        try:
            processor.process(f"data/video/video_{i}.webm", audio_path)
        except:
            prev_url = None
            continue

        audio_features = audio_model.extract_from_audio_folder("output/split_audio").to("cpu").squeeze() # X_A [ N_U, 768 ]

        # Save audio features
        torch.save(audio_features, audio_features_path)

        video_features = video_model.extract_from_video(frame_path).to("cpu").squeeze()  # X_V [ N_U, 768 ]

        # Save video features
        torch.save(video_features, video_features_path)

        emb, graph = relation_constructor.get_relation_graph("output/split_audio/utterance.csv","output/utterance.pt","output/audio_features.pt","output/video_features.pt")

        # 그래프 데이터 저장
        torch.save(emb, f"data/embedding/emb_{i+1:04d}.pt")
        torch.save(graph, f"data/graph/graph_{i+1:04d}.pt")

        emb_file_list.append(f"data/embedding/emb_{i+1:04d}.pt")
        graph_file_list.append(f"data/graph/graph_{i+1:04d}.pt")
        title_list.append(title)
        url_list.append(url)
        # question_list.append(question)

        prev_url = url
        prev_i = i

    # Save the list of files
    dfdf = {
        "title":title_list,
        "url":url_list,
        # "question":question_list,
        "emb_file":emb_file_list,
        "graph_file":graph_file_list
    }
    
    df_samples_file = pd.DataFrame(dfdf)
    df_samples_file.to_csv("data/data_annotation_pe.csv", index=False)

    print("모든 비디오 처리 완료.")


#######################################


    # video_path = "video.webm"
    # audio_path = "output/split_audio"
    # frame_path = "output/extracted_frames"
    # audio_features_path = "/media/user/data/polymarket/data/embbeding/audio/"
    # video_features_path = "/media/user/data/polymarket/data/embbeding/video/"
    # text_features_path = "/media/user/data/polymarket/data/embbeding/text/"
    # question_path = "/media/user/data/polymarket/data/embbeding/question/"

    # os.makedirs(audio_features_path, exist_ok=True)
    # os.makedirs(video_features_path, exist_ok=True)
    # os.makedirs(text_features_path, exist_ok=True)
    # os.makedirs(question_path, exist_ok=True)

    # processor = NewsAudioVideoProcessor()
    # audio_model = AudioFeatureExtractor()
    # video_model = VideoFeatureExtractor()
    # text_model = TextFeatureExtractor()
    # # relation_constructor = RelationConstructor()

    # df_samples = pd.read_csv("matching_questions_politics_economy_concat.csv")
    # df_samples.sort_values(by=['url'], inplace=True)
    # df_annotation = {}

    # title_list = []
    # url_list = []
    # question_list = []
    # text_file_list = []
    # audio_file_list = []
    # video_file_list = []
    # question_file_list = []
    # # graph_file_list = []

    # prev_url = None
    # prev_i = None

    # for i in tqdm(range(len(df_samples)),desc="Data Processing"):
    #     url = df_samples.iloc[i]['url']
    #     title = df_samples.iloc[i]['title']
    #     question = df_samples.iloc[i]['matching_questions']

    #     if prev_url != None and prev_url == url:
    #         title_list.append(title)
    #         url_list.append(url)
    #         question_list.append(question)
    #         text_file_list.append(f"{text_features_path}text_{prev_i+1:04d}.pt")
    #         audio_file_list.append(f"{audio_features_path}audio_{prev_i+1:04d}.pt")
    #         video_file_list.append(f"{video_features_path}video_{prev_i+1:04d}.pt")

    #         question_features = text_model.encode(df_samples.iloc[i]['matching_questions']).to("cpu").squeeze()  # X_Q [ N_U, 768 ]
            
    #         torch.save(question_features, f"{question_path}question_{i+1:04d}.pt")
    #         question_file_list.append(f"{question_path}question_{i+1:04d}.pt")
            
    #         continue

    #     exist = os.path.exists(f"/media/user/data/polymarket/data/video/video_{i}.webm")
    #     if exist:
    #         print(f"이미 다운로드된 비디오입니다. URL: {url}")
    #         title_list.append(title)
    #         url_list.append(url)
    #         question_list.append(question)
    #         text_file_list.append(f"{text_features_path}text_{i+1:04d}.pt")
    #         audio_file_list.append(f"{audio_features_path}audio_{i+1:04d}.pt")
    #         video_file_list.append(f"{video_features_path}video_{i+1:04d}.pt")

    #         # question_features = text_model.encode(df_samples.iloc[i]['matching_questions']).to("cpu").squeeze()  # X_Q [ N_U, 768 ]
            
    #         # torch.save(question_features, f"{question_path}question_{i+1:04d}.pt")
    #         question_file_list.append(f"{question_path}question_{i+1:04d}.pt")
    #         prev_url = url
    #         prev_i = i
    #         continue
    #     else:
    #         dur = download_video(url, "/media/user/data/polymarket/data/video", i)

    #         if dur > 3600:
    #             print(f"Error!!. URL: {url}")
    #             prev_url = None
    #             continue

    #     try:

    #         processor.process(f"/media/user/data/polymarket/data/video/video_{i}.webm", f"/media/user/data/polymarket/data/audio/audio.wav")

    #         audio_features = audio_model.extract_from_audio_folder(audio_path).to("cpu").squeeze() # X_A [ N_U, 768 ]

    #         # Save audio features
    #         # torch.save(audio_features, audio_features_path)

    #         video_features = video_model.extract_from_video(frame_path).to("cpu").squeeze()  # X_V [ N_U, 768 ]

    #         # Save video features
    #         # torch.save(video_features, video_features_path)

    #         # Save text features
    #         text_features = text_model.encode(df_samples.iloc[i]['text']).to("cpu").squeeze()  # X_T [ N_U, 768 ]

    #         # emb, graph = relation_constructor.get_relation_graph("output/split_audio/utterance.csv","output/utterance.pt","output/audio_features.pt","output/video_features.pt")

    #         question_features = text_model.encode(df_samples.iloc[i]['matching_questions']).to("cpu").squeeze()  # X_Q [ N_U, 768 ]
            
    #         # 데이터 저장
    #         torch.save(audio_features, f"{audio_features_path}audio_{i+1:04d}.pt")
    #         torch.save(video_features, f"{video_features_path}video_{i+1:04d}.pt")
    #         torch.save(text_features, f"{text_features_path}text_{i+1:04d}.pt")
    #         torch.save(question_features, f"{question_path}question_{i+1:04d}.pt")

    #         title_list.append(title)
    #         url_list.append(url)
    #         question_list.append(question)
    #         text_file_list.append(f"{text_features_path}text_{i+1:04d}.pt")
    #         audio_file_list.append(f"{audio_features_path}audio_{i+1:04d}.pt")
    #         video_file_list.append(f"{video_features_path}video_{i+1:04d}.pt")
    #         question_file_list.append(f"{question_path}question_{i+1:04d}.pt")

    #     except:
    #         prev_url = None
    #         continue

    #     prev_url = url
    #     prev_i = i

    # # Save the list of files
    # df_annotation["title"] = title_list
    # df_annotation["url"] = url_list
    # df_annotation["matching_questions"] = question_list
    # df_annotation["audio_file"] = audio_file_list
    # df_annotation["video_file"] = video_file_list
    # df_annotation["question_file"] = question_file_list
    # df_annotation["text_file"] = text_file_list
    # pd.DataFrame(df_annotation).to_csv("/media/user/data/polymarket/data/data_annotation_politics_economy.csv", index=False)

    # print("모든 비디오 처리 완료.")