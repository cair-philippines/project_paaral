"""Create the PAARAL ERD on a Miro board via REST API v2.

Run: python3 docs/gen_miro_erd.py
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
        print(
            f"  ERROR {r.status_code} {method.upper()}"
            f" {path}: {r.text[:300]}"
        )
        r.raise_for_status()
    time.sleep(0.2)
    return r.json()


def mk_shape(bid, html, fill, color, border, cx, cy, w, h,
             fs=12, align="left", valign="top", bw=1):
    """Create a styled rectangle shape on a Miro board.

    Parameters
    ----------
    bid : str
        Miro board ID.
    html : str
        HTML content for the shape label.
    fill : str
        Fill colour as hex string.
    color : str
        Text colour as hex string.
    border : str
        Border colour as hex string.
    cx, cy : int
        Centre position in Miro coordinates.
    w, h : int
        Width and height in pixels.
    fs : int
        Font size.
    align : str
        Horizontal text alignment.
    valign : str
        Vertical text alignment.
    bw : int
        Border width.

    Returns
    -------
    dict
        Miro API response for the created shape.
    """
    return call("post", f"/boards/{bid}/shapes", {
        "data":     {"shape": "rectangle", "content": html},
        "style":    {"fillColor": fill, "color": color,
                     "borderColor": border,
                     "borderWidth": str(bw), "fontSize": str(fs),
                     "textAlign": align, "textAlignVertical": valign,
                     "fontFamily": "open_sans"},
        "geometry": {"width": w, "height": h},
        "position": {"x": cx, "y": cy, "origin": "center"},
    })


def mk_text(bid, html, cx, cy, w, fs=16, color="#1a1d23"):
    """Create a text widget on a Miro board.

    Parameters
    ----------
    bid : str
        Miro board ID.
    html : str
        HTML content.
    cx, cy : int
        Centre position in Miro coordinates.
    w : int
        Width in pixels.
    fs : int
        Font size.
    color : str
        Text colour as hex string.

    Returns
    -------
    dict
        Miro API response for the created text widget.
    """
    return call("post", f"/boards/{bid}/texts", {
        "data":     {"content": html},
        "style":    {"color": color, "fontSize": str(fs),
                     "textAlign": "center", "fontFamily": "open_sans"},
        "geometry": {"width": w},
        "position": {"x": cx, "y": cy, "origin": "center"},
    })


def mk_connector(bid, a, b, label=""):
    """Draw a connector between two Miro items.

    Parameters
    ----------
    bid : str
        Miro board ID.
    a : str
        ID of the start item.
    b : str
        ID of the end item.
    label : str, optional
        Caption shown at the connector midpoint.

    Returns
    -------
    dict
        Miro API response for the created connector.
    """
    d = {
        "startItem": {"id": a, "snapTo": "auto"},
        "endItem":   {"id": b, "snapTo": "auto"},
        "style":     {"strokeColor": "#94a3b8", "strokeWidth": "2",
                      "endStrokeCap": "arrow", "startStrokeCap": "none"},
    }
    if label:
        d["captions"] = [{"content": label, "position": "50"}]
    return call("post", f"/boards/{bid}/connectors", d)


# ── theme ────────────────────────────────────────────────────────────────────
V = {
    "student": {"accent": "#1a4b8c", "bg": "#dbeaff"},
    "core":    {"accent": "#92400e", "bg": "#ffedd5"},
    "school":  {"accent": "#16a34a", "bg": "#dcfce7"},
    "deped":   {"accent": "#dc2626", "bg": "#fee2e2"},
}

# ── entities: (view, [(name, "PK"|"FK"|"", type_str)]) ───────────────────────
ENTITIES = {
    "SCHOOL": ("core", [
        ("id",                "PK", "string"),
        ("name",              "",   "string"),
        ("type",              "",   "enum  public|private_esc|private_no_esc"),
        ("sector",            "",   "enum  sectarian|non_sectarian"),
        ("region",            "",   "string"),
        ("province",          "",   "string"),
        ("municipality",      "",   "string"),
        ("barangay",          "",   "string"),
        ("postal_code",       "",   "string"),
        ("lat / lng",         "",   "decimal"),
        ("tuition",           "",   "int  ₱ annual"),
        ("esc_subsidy",       "",   "int  ₱ annual"),
        ("net_cost",          "",   "int  tuition − subsidy"),
        ("slots_total",       "",   "int"),
        ("slots_available",   "",   "int"),
        ("distance_km",       "",   "decimal"),
        ("commute_minutes",   "",   "int"),
        ("esc_rating",        "",   "int  0–5"),
        ("admission_category", "",  "string"),
    ]),
    "LEARNER": ("student", [
        ("lrn",              "PK", "string  12-digit"),
        ("given_name",       "",   "string"),
        ("family_name",      "",   "string"),
        ("barangay",         "",   "string"),
        ("municipality",     "",   "string"),
        ("grade6_school_id", "FK", "→ SCHOOL"),
    ]),
    "APPLICATION": ("student", [
        ("id",           "PK", "uuid"),
        ("lrn",          "FK", "→ LEARNER"),
        ("status",       "",   "enum  draft|submitted|cancelled"),
        ("submitted_at", "",   "timestamp"),
    ]),
    "WISHLIST_ENTRY": ("student", [
        ("id",             "PK", "uuid"),
        ("application_id", "FK", "→ APPLICATION"),
        ("school_id",      "FK", "→ SCHOOL"),
        ("rank",           "",   "int  1 = first choice"),
    ]),
    "ELIGIBILITY_ASSESSMENT": ("student", [
        ("id",             "PK", "uuid"),
        ("application_id", "FK", "→ APPLICATION"),
        ("esc_intent",     "",   "enum  yes|no"),
        ("school_type",    "",   "enum  public|private|als"),
        ("segs",           "",   "json  4ps,gidca,ip,pwd,special,cbms"),
        ("income",         "",   "enum  poor|low|lower_middle|middle|above"),
        ("employment",     "",   "enum  local|abroad|business|unemployed"),
        ("category",       "",   "enum  A|B|C|D|none"),
    ]),
    "SURVEY_RESPONSE": ("student", [
        ("id",             "PK", "uuid"),
        ("application_id", "FK", "→ APPLICATION"),
        ("ease",           "",   "int  1–5"),
        ("helpful",        "",   "enum  yes|somewhat|no"),
        ("concern",        "",   "enum  cost|distance|quality|slots"),
        ("suggestions",    "",   "text  optional"),
    ]),
    "DOCUMENT_UPLOAD": ("student", [
        ("id",             "PK", "uuid"),
        ("application_id", "FK", "→ APPLICATION"),
        ("doc_type",       "",   "string"),
        ("storage_path",   "",   "string"),
        ("uploaded_at",    "",   "timestamp"),
    ]),
    "SCHOOL_USER": ("school", [
        ("id",        "PK", "uuid"),
        ("school_id", "FK", "→ SCHOOL"),
        ("name",      "",   "string"),
        ("role",      "",   "enum  staff|esc_committee_member"),
    ]),
    "APPLICATION_REVIEW": ("school", [
        ("id",             "PK", "uuid"),
        ("application_id", "FK", "→ APPLICATION"),
        ("school_id",      "FK", "→ SCHOOL"),
        ("reviewed_by",    "FK", "→ SCHOOL_USER"),
        ("decision",       "",   "enum  pending|approved|waitlisted|rejected"),
        ("remarks",        "",   "text"),
        ("reviewed_at",    "",   "timestamp"),
    ]),
    "SLOT_ALLOCATION": ("school", [
        ("id",              "PK", "uuid"),
        ("school_id",       "FK", "→ SCHOOL"),
        ("school_year",     "",   "string  e.g. 2026-2027"),
        ("slots_allocated", "",   "int"),
        ("slots_consumed",  "",   "int"),
        ("status",          "",   "enum  open|closed|full"),
    ]),
    "REGION": ("deped", [
        ("code", "PK", "string"),
        ("name", "",   "string"),
    ]),
    "DIVISION": ("deped", [
        ("code",        "PK", "string"),
        ("name",        "",   "string"),
        ("region_code", "FK", "→ REGION"),
    ]),
    "PLANNING_SNAPSHOT": ("deped", [
        ("id",                  "PK", "uuid"),
        ("region_code",         "FK", "→ REGION"),
        ("school_year",         "",   "string"),
        ("projected_g7_enroll", "",   "int"),
        ("public_jhs_capacity", "",   "int"),
        ("esc_slots_needed",    "",   "int"),
        ("esc_slots_available", "",   "int"),
        ("generated_at",        "",   "timestamp"),
    ]),
}

RELS = [
    ("LEARNER",              "APPLICATION",           "submits  1→0..*"),
    ("LEARNER",              "SCHOOL",                "attended Gr6"),
    ("APPLICATION",          "WISHLIST_ENTRY",        "contains  1→1..*"),
    ("APPLICATION",          "ELIGIBILITY_ASSESSMENT", "has  1→0..1"),
    ("APPLICATION",          "SURVEY_RESPONSE",       "has  1→0..1"),
    ("APPLICATION",          "DOCUMENT_UPLOAD",       "attaches  1→0..*"),
    ("APPLICATION",          "APPLICATION_REVIEW",    "reviewed via  1→0..*"),
    ("SCHOOL",               "WISHLIST_ENTRY",        "appears in  1→0..*"),
    ("SCHOOL",               "SLOT_ALLOCATION",       "has  1→0..*"),
    ("SCHOOL",               "APPLICATION_REVIEW",    "receives  1→0..*"),
    ("SCHOOL",               "DIVISION",              "governed by  0..*→1"),
    ("SCHOOL_USER",          "SCHOOL",                "belongs to  0..*→1"),
    ("SCHOOL_USER",          "APPLICATION_REVIEW",    "makes  1→0..*"),
    ("REGION",               "DIVISION",              "contains  1→1..*"),
    ("REGION",               "PLANNING_SNAPSHOT",     "analyzed in  1→0..*"),
]

# ── layout ───────────────────────────────────────────────────────────────────
EW   = 280
HDRH = 44
ROWH = 27
BPAD = 14
VGAP = 65

COLUMNS = [
    (0, ["LEARNER", "APPLICATION", "WISHLIST_ENTRY"]),
    (1, ["ELIGIBILITY_ASSESSMENT", "SURVEY_RESPONSE", "DOCUMENT_UPLOAD"]),
    (2, ["SCHOOL"]),
    (3, ["SCHOOL_USER", "APPLICATION_REVIEW", "SLOT_ALLOCATION"]),
    (4, ["REGION", "DIVISION", "PLANNING_SNAPSHOT"]),
]
COL_X      = {0: -1700, 1: -1020, 2: -310, 3: 420, 4: 1100}
ENTITY_COL = {k: ci for ci, ks in COLUMNS for k in ks}


def bh(n):
    """Return body height for n field rows.

    Parameters
    ----------
    n : int
        Number of field rows.

    Returns
    -------
    int
        Body height in pixels.
    """
    return n * ROWH + BPAD


def th(n):
    """Return total entity height (header + body) for n field rows.

    Parameters
    ----------
    n : int
        Number of field rows.

    Returns
    -------
    int
        Total height in pixels.
    """
    return HDRH + bh(n)


TOP_Y = {}
for ci, keys in COLUMNS:
    y = -850
    for k in keys:
        TOP_Y[k] = y
        y += th(len(ENTITIES[k][1])) + VGAP


def field_html(fields):
    """Build HTML markup for the field rows of an entity shape.

    Parameters
    ----------
    fields : list
        List of (name, tag, typ) tuples where tag is 'PK', 'FK', or ''.

    Returns
    -------
    str
        Concatenated HTML paragraphs, one per field.
    """
    rows = []
    for name, tag, typ in fields:
        if tag == "PK":
            rows.append(f"<p><strong>◆ {name}</strong>  {typ}</p>")
        elif tag == "FK":
            rows.append(f"<p><em>◇ {name}</em>  {typ}</p>")
        else:
            rows.append(f"<p>  {name}  {typ}</p>")
    return "".join(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    """Create the PAARAL ERD board on Miro and print the board URL."""
    print("Creating board...")
    board = call("post", "/boards", {
        "name": "PAARAL — Entity-Relationship Diagram",
        "description": (
            "ERD covering Student View, School View,"
            " and DepEd Planning View."
        ),
    })
    bid = board["id"]
    url = f"https://miro.com/app/board/{bid}/"
    print(f"  ✓ {url}")

    # title
    mk_text(
        bid,
        "<p><strong>PAARAL — Entity-Relationship Diagram</strong></p>",
        -310, -1140, 2600, fs=24, color="#1a1d23",
    )

    # section labels
    for label, cx, cy, w, v_key in [
        ("STUDENT VIEW", -1360, -970, 1000, "student"),
        ("CORE",          -310, -970,  310, "core"),
        ("SCHOOL VIEW",    420, -970,  310, "school"),
        ("DEPED VIEW",    1100, -970,  310, "deped"),
    ]:
        mk_shape(bid, f"<p><strong>{label}</strong></p>",
                 V[v_key]["bg"], V[v_key]["accent"], V[v_key]["accent"],
                 cx, cy, w, 38, fs=13, align="center",
                 valign="middle", bw=2)

    # entities
    hdr_ids = {}
    for key, (vtag, fields) in ENTITIES.items():
        v   = V[vtag]
        cx  = COL_X[ENTITY_COL[key]]
        top = TOP_Y[key]
        fh  = bh(len(fields))

        hdr = mk_shape(bid, f"<p><strong>{key}</strong></p>",
                       v["accent"], "#ffffff", v["accent"],
                       cx, top + HDRH / 2, EW, HDRH,
                       fs=13, align="left", valign="middle", bw=2)
        hdr_ids[key] = hdr["id"]
        print(f"  {key}")

        mk_shape(bid, field_html(fields),
                 "#ffffff", "#374151", v["accent"],
                 cx, top + HDRH + fh / 2, EW, fh,
                 fs=11, align="left", valign="top", bw=1)

    # connectors
    print("Drawing connectors...")
    for a, b, label in RELS:
        mk_connector(bid, hdr_ids[a], hdr_ids[b], label)
        print(f"  {a} → {b}")

    print(f"\n✓ Board ready: {url}")


if __name__ == "__main__":
    main()
