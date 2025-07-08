import requests
import json
import networkx as nx
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON


# 특정 날짜 기준 리비전 내용 가져오기
def fetch_revision_claims(entity_id: str, date: str):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": entity_id,
        "rvlimit": 1,
        "rvend": f"{date}T00:00:00Z",
        "rvdir": "older",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": 2
    }
    r = requests.get(url, params=params)
    data = r.json()
    try:
        content = data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        content_json = json.loads(content)
        claims = content_json.get("claims", {})
        linked_qids = set()

        for prop in claims:
            for claim in claims[prop]:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value")

                # 연결된 Q-ID 추출 (wikibase-entityid)
                if isinstance(value, dict) and "id" in value:
                    linked_qids.add(value["id"])

        return linked_qids
    except Exception as e:
        print(f"리비전 파싱 오류: {e}")
        return set()