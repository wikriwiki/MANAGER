#!/usr/bin/env python3
"""
fetch_pid_labels.py
-------------------
● `claims` 테이블에 등장하는 **모든 PID(property ID)** 를 모아
  Wikidata API로 영어 라벨을 가져와서
  같은 SQLite DB 안의 **property_labels** 테이블에 저장합니다.

$ python fetch_pid_labels.py data/wiki_revision.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

API_URL = "https://www.wikidata.org/w/api.php"


# --------------------------------------------------------------------------- #
# requests.Session with retry/back-off                                        #
# --------------------------------------------------------------------------- #
def init_session(retries: int = 4, backoff: float = 0.3) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.headers.update(
        {
            "User-Agent": (
                "PIDLabelCrawler/1.0 "
                "(https://github.com/your-handle; mailto:your@email)"
            )
        }
    )
    return sess


SESSION = init_session()

# --------------------------------------------------------------------------- #
# SQLite helpers                                                              #
# --------------------------------------------------------------------------- #
SCHEMA_EXT = """
CREATE TABLE IF NOT EXISTS property_labels (
    pid   TEXT PRIMARY KEY,
    label TEXT
);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_EXT)
    return conn


# --------------------------------------------------------------------------- #
# Wikidata API wrapper                                                        #
# --------------------------------------------------------------------------- #
def fetch_property_label(pid: str) -> str:
    """Return English label for a PID (empty string if missing)."""
    params = {
        "action": "wbgetentities",
        "ids": pid,
        "format": "json",
        "languages": "en",
        "props": "labels",
    }
    try:
        data = SESSION.get(API_URL, params=params, timeout=20).json()
        return data["entities"][pid]["labels"]["en"]["value"]
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Main logic                                                                  #
# --------------------------------------------------------------------------- #
def crawl_pids(db_path: str | Path, sleep: float = 0.1):
    conn = init_db(db_path)
    cur = conn.cursor()

    # 1) 이미 저장된 PID 집합
    cur.execute("SELECT pid FROM property_labels")
    existing: set[str] = {row[0] for row in cur.fetchall()}

    # 2) claims 에서 PID 전부 수집
    cur.execute("SELECT DISTINCT property FROM claims")
    all_pids: set[str] = {row[0] for row in cur.fetchall()}

    # 3) 아직 없는 PID 만
    targets = sorted(all_pids - existing)
    if not targets:
        logging.info("No new PID to crawl. Done.")
        return

    for pid in tqdm(targets, desc="Fetching PID labels"):
        label = fetch_property_label(pid)
        cur.execute(
            "INSERT OR IGNORE INTO property_labels (pid, label) VALUES (?, ?)",
            (pid, label),
        )
        time.sleep(sleep)

    conn.commit()
    conn.close()
    logging.info("Inserted %d new property labels.", len(targets))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl PID labels into SQLite DB")
    p.add_argument("db", type=Path, help="Path to existing wiki_revision.db")
    p.add_argument("--sleep", type=float, default=0.1, help="Sleep between API calls (s)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.db.exists():
        logging.error("DB file not found: %s", args.db)
        sys.exit(1)

    crawl_pids(args.db, sleep=args.sleep)


if __name__ == "__main__":
    main()
