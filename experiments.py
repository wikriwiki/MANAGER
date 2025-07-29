# experiments.py
import sqlite3
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse
def get_git_commit_hash() -> Optional[str]:
    """현재 Git 커밋 해시를 가져옵니다."""
    try:
        # 현재 스크립트가 Git 리포지토리 내에 있는지 확인
        # 현재 작업 디렉토리가 Git 리포지토리 루트라고 가정
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.getcwd()).strip().decode('ascii')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def init_experiment_db(db_path: str = "experiments.db") -> sqlite3.Connection:
    """
    실험 결과를 저장할 SQLite 데이터베이스를 초기화하고 연결을 반환합니다.
    테이블이 없으면 생성합니다.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # experiments 테이블: 각 실험 실행의 전체 정보 및 메트릭 저장
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            data_db_path TEXT,
            cache_dir TEXT,
            frames_dir TEXT,
            wav_dir TEXT,
            epochs INTEGER,
            learning_rate REAL,
            random_seed INTEGER,
            max_samples INTEGER,                      
            val_f1 REAL,
            val_precision REAL,
            val_recall REAL,
            test_f1 REAL,
            test_precision REAL,
            test_recall REAL,
            test_roc_auc REAL,
            checkpoint_path TEXT,
            notes TEXT,
            git_commit_hash TEXT
        );
    """)

    # experiment_samples 테이블: 각 실험 내 개별 샘플의 예측 결과 저장
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,               -- 'experiments' 테이블의 id와 연결
            sample_video_id TEXT NOT NULL,                -- 해당 샘플의 video_id
            sample_person_name TEXT NOT NULL,             -- 해당 샘플의 person_name
            data_split TEXT NOT NULL,                     -- 'val' 또는 'test' (어떤 데이터셋이었는지)
            true_label INTEGER NOT NULL,                  -- 실제 레이블 (0 또는 1)
            predicted_label INTEGER NOT NULL,             -- 예측된 레이블 (0 또는 1)
            prediction_score REAL NOT NULL,               -- 모델의 예측 스코어 (sigmoid 출력 값)
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        );
    """)
    conn.commit()
    return conn

def insert_new_experiment(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """새로운 실험 시작 시 기본 정보를 experiments 테이블에 삽입하고 experiment_id를 반환합니다."""
    cur = conn.cursor()
    
    git_hash = get_git_commit_hash()
    
    cur.execute("""
        INSERT INTO experiments (
            model_name, data_db_path, cache_dir, frames_dir, wav_dir,
            epochs, learning_rate, random_seed, max_samples, git_commit_hash, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "GraphTokenManager", # 모델 이름 고정
        args.db, str(args.cache), str(args.frames), str(args.wav), # Path 객체를 str로 변환
        args.epochs, args.lr, args.seed, args.max_samples, git_hash,
        f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ))
    conn.commit()
    return cur.lastrowid # 새로 삽입된 레코드의 ID 반환

def update_experiment_metrics(conn: sqlite3.Connection, exp_id: int, split: str, metrics: dict):
    """
    지정된 실험 ID의 experiments 테이블에 성능 지표를 업데이트합니다.
    split 인자('val' 또는 'test')에 따라 해당 컬럼을 업데이트합니다.
    """
    cur = conn.cursor()
    if split == "val":
        cur.execute("""
            UPDATE experiments
            SET val_f1 = ?, val_precision = ?, val_recall = ?
            WHERE id = ?
        """, (metrics['f1'], metrics['precision'], metrics['recall'], exp_id))
    elif split == "test":
        cur.execute("""
            UPDATE experiments
            SET test_f1 = ?, test_precision = ?, test_recall = ?, test_roc_auc = ?
            WHERE id = ?
        """, (metrics['f1'], metrics['precision'], metrics['recall'], metrics['roc_auc'], exp_id))
    conn.commit()

def insert_sample_predictions(conn: sqlite3.Connection, exp_id: int, predictions: list, data_split: str):
    """
    개별 샘플 예측 결과를 experiment_samples 테이블에 삽입합니다.
    predictions 리스트는 각 요소가 (video_id, person_name, true_label, predicted_label, prediction_score) 형태의 튜플이어야 합니다.
    """
    cur = conn.cursor()
    data_to_insert = [
        (exp_id, p[0], p[1], data_split, p[2], p[3], p[4]) # video_id, person_name, data_split, true_label, predicted_label, prediction_score
        for p in predictions
    ]
    cur.executemany("""
        INSERT INTO experiment_samples (
            experiment_id, sample_video_id, sample_person_name, data_split,
            true_label, predicted_label, prediction_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    conn.commit()