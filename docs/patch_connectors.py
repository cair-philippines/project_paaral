"""Add connectors to the existing PAARAL ERD board.

Board: https://miro.com/app/board/uXjVHPVhBsk=/
"""

import time

import requests

TOKEN = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_kTaxxIJFLvUyOs_Y3PmhoR_3omA"
BASE  = "https://api.miro.com/v2"
HDRS  = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
BID   = "uXjVHPVhBsk="


def call(method, path, data=None):
    """Send an HTTP request to the Miro API and return JSON.

    Parameters
    ----------
    method : str
        HTTP method name (e.g. 'get', 'post').
    path : str
        API path relative to BASE.
    data : dict, optional
        JSON body payload.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    r = getattr(requests, method)(
        f"{BASE}{path}", headers=HDRS, json=data
    )
    if not r.ok:
        print(f"  ERROR {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    time.sleep(0.2)
    return r.json()

# Fetch all shapes on the board to find entity header IDs by their content
print("Fetching board items...")
items = []
cursor = None
while True:
    params = {"limit": 50, "type": "shape"}
    if cursor:
        params["cursor"] = cursor
    r = requests.get(f"{BASE}/boards/{BID}/items", headers=HDRS, params=params)
    r.raise_for_status()
    data = r.json()
    items.extend(data.get("data", []))
    cursor = data.get("cursor")
    if not cursor:
        break
    time.sleep(0.2)

print(f"  Found {len(items)} shapes")

# Find header shapes — they have bold entity names as content
# Headers have fillColor matching one of our accent colors
ACCENT_COLORS = {"#1a4b8c", "#92400e", "#16a34a", "#dc2626"}
ENTITY_NAMES = [
    "SCHOOL", "LEARNER", "APPLICATION", "WISHLIST_ENTRY",
    "ELIGIBILITY_ASSESSMENT", "SURVEY_RESPONSE", "DOCUMENT_UPLOAD",
    "SCHOOL_USER", "APPLICATION_REVIEW", "SLOT_ALLOCATION",
    "REGION", "DIVISION", "PLANNING_SNAPSHOT",
]

hdr_ids = {}
for item in items:
    style = item.get("style", {})
    fill  = style.get("fillColor", "").lower()
    if fill in ACCENT_COLORS:
        content = item.get("data", {}).get("content", "")
        for name in ENTITY_NAMES:
            if name in content and name not in hdr_ids:
                hdr_ids[name] = item["id"]
                print(f"  matched {name} → {item['id']}")
                break

print(f"\nMatched {len(hdr_ids)}/{len(ENTITY_NAMES)} entity headers")
missing = [n for n in ENTITY_NAMES if n not in hdr_ids]
if missing:
    print(f"  Missing: {missing}")

RELS = [
    ("LEARNER",              "APPLICATION",           "submits  1→0..*"),
    ("LEARNER",              "SCHOOL",                "attended Gr6"),
    ("APPLICATION",          "WISHLIST_ENTRY",        "contains  1→1..*"),
    ("APPLICATION",          "ELIGIBILITY_ASSESSMENT","has  1→0..1"),
    ("APPLICATION",          "SURVEY_RESPONSE",       "has  1→0..1"),
    ("APPLICATION",          "DOCUMENT_UPLOAD",       "attaches  1→0..*"),
    ("APPLICATION",          "APPLICATION_REVIEW",    "reviewed via  1→0..*"),
    ("SCHOOL",               "WISHLIST_ENTRY",         "appears in  1→0..*"),
    ("SCHOOL",               "SLOT_ALLOCATION",        "has  1→0..*"),
    ("SCHOOL",               "APPLICATION_REVIEW",     "receives  1→0..*"),
    ("SCHOOL",               "DIVISION",               "governed by  0..*→1"),
    ("SCHOOL_USER",          "SCHOOL",                 "belongs to  0..*→1"),
    ("SCHOOL_USER",          "APPLICATION_REVIEW",     "makes  1→0..*"),
    ("REGION",               "DIVISION",               "contains  1→1..*"),
    ("REGION",               "PLANNING_SNAPSHOT",      "analyzed in  1→0..*"),
]

print("\nDrawing connectors...")
for a, b, label in RELS:
    if a not in hdr_ids or b not in hdr_ids:
        print(f"  SKIP {a} → {b} (id missing)")
        continue
    call("post", f"/boards/{BID}/connectors", {
        "startItem": {"id": hdr_ids[a], "snapTo": "auto"},
        "endItem":   {"id": hdr_ids[b], "snapTo": "auto"},
        "captions":  [{"content": label, "position": "50%"}],
        "style":     {"strokeColor": "#94a3b8", "strokeWidth": "2",
                      "endStrokeCap": "arrow", "startStrokeCap": "none"},
    })
    print(f"  ✓ {a} → {b}")

print("\n✓ Done — https://miro.com/app/board/uXjVHPVhBsk=/")
