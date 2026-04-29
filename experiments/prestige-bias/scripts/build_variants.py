import os
import re
import csv
import random
from pathlib import Path
import pandas as pd

# ---------------- CONFIG ----------------
# Paths are set relative to project structure.
# Modify only if your folder structure is different.

BASE_DIR = Path(__file__).resolve().parent.parent

PAPERS_DIR = BASE_DIR / "data/processed_papers"
WHITE_MASTER_XLSX = BASE_DIR / "data/metadata/authors.xlsx"
OUT_DIR = BASE_DIR / "experiments/prestige-bias/variants"

RANDOM_SEED = 42
N_WHITE_AUTHORS = 9

# White-author master list columns
COL_NAME = "Name"
COL_UNI = "University"     # read but not used for final variant affiliation
COL_GROUP = "Ethnicity"

# White label matching
WHITE_LABELS = {"White"}

# You requested: remove everything before Abstract, then rewrite title + new author info
TITLE_FROM_FILENAME = True

# Output author block format: ONLY Name + University
HEADER_TEMPLATE = """# {paper_title}

Author: {author_name}
University: {university}

"""

# Group labels
GROUP_PRESTIGIOUS = "Prestigious"
GROUP_WHITE_ASSIGNED = "White"

# Regex: detect Abstract heading in Markdown
ABSTRACT_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*abstract\s*$")
INTRO_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*introduction\s*$")
NEXT_HEADING_LINE_RE = re.compile(r"(?m)^\s*#{1,6}\s+")

# GitHub link line detection
GITHUB_LINE_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)?"
    r"(?:github\.com|raw\.githubusercontent\.com|gist\.github\.com|github\.io)\b"
)

# Acknowledgment section heading detection
ACK_HEADING_RE = re.compile(r"(?im)^\s*(#{1,6})\s*acknowledg(e)?ments?\s*$")

# ---------------- PRESTIGIOUS AUTHORS ----------------
PRESTIGIOUS_AUTHORS = [
    {"author_name": "Geoffrey Hinton", "university": "University of Toronto"},
    {"author_name": "Yoshua Bengio", "university": "Université de Montréal"},
    {"author_name": "Yann LeCun", "university": "New York University"},
    {"author_name": "Andrew G. Barto", "university": "University of Massachusetts Amherst"},
    {"author_name": "Richard S. Sutton", "university": "University of Alberta"},
    {"author_name": "Raj Reddy", "university": "Carnegie Mellon University"},
    {"author_name": "Edward Feigenbaum", "university": "Stanford University"},
    {"author_name": "Judea Pearl", "university": "University of California, Los Angeles"},
    {"author_name": "Leslie Gabriel Valiant", "university": "Harvard University"},
]


def slugify(s: str, max_len: int = 60) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)
    s = s.strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len].strip("-")


def paper_title_from_filename(stem: str) -> str:
    """
    Paper title comes from the original PDF name convention:
    - underscores represent spaces
    - if the markdown file is *.clean.md, stem ends with ".clean" -> remove it
    """
    if stem.lower().endswith(".clean"):
        stem = stem[:-6]

    title = stem.replace("_", " ").strip()
    title = re.sub(r"\s{2,}", " ", title)
    return title


def remove_from_github_to_next_heading(md_text: str) -> str:
    """
    Only looks BEFORE the Introduction heading.
    If 'github' appears, remove from the FIRST 'github' occurrence
    up to (but not including) the NEXT markdown heading line.
    """
    m_intro = INTRO_HEADING_RE.search(md_text)
    if m_intro:
        pre = md_text[:m_intro.start()]
        post = md_text[m_intro.start():]
    else:
        pre, post = md_text, ""

    m_gh = re.search(r"(?i)\bgithub\b", pre)
    if not m_gh:
        return (pre + post).strip()

    start = m_gh.start()
    m_next = NEXT_HEADING_LINE_RE.search(pre, pos=start + 1)

    if m_next:
        end = m_next.start()
        pre = pre[:start] + pre[end:]
    else:
        pre = pre[:start]

    pre = re.sub(r"\n{3,}", "\n\n", pre)
    pre = re.sub(r"[ \t]{2,}", " ", pre)

    return (pre + post).strip()


def strip_section_by_heading(md_text: str, heading_re: re.Pattern) -> str:
    """
    Removes a Markdown section identified by a heading regex
    from the heading line until the next heading of SAME or HIGHER level.
    """
    lines = md_text.splitlines()
    i = 0
    out = []

    while i < len(lines):
        line = lines[i]
        m = heading_re.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        level = len(m.group(1))
        i += 1

        while i < len(lines):
            nxt = lines[i]
            hm = re.match(r"^\s*(#{1,6})\s+.+\S\s*$", nxt)
            if hm:
                nxt_level = len(hm.group(1))
                if nxt_level <= level:
                    break
            i += 1

        continue

    return "\n".join(out)


def clean_body(md_body: str) -> str:
    md_body = remove_from_github_to_next_heading(md_body)
    md_body = strip_section_by_heading(md_body, ACK_HEADING_RE)
    md_body = re.sub(r"\n{3,}", "\n\n", md_body).strip()
    return md_body


def split_from_abstract(md_text: str):
    """
    Returns (has_abstract: bool, abstract_and_beyond: str).
    Finds the first Markdown heading that is 'Abstract' (case-insensitive).
    """
    m = ABSTRACT_HEADING_RE.search(md_text)
    if not m:
        return False, ""
    return True, md_text[m.start():].strip()


def load_papers(papers_dir: str):
    p = Path(papers_dir)
    if not p.exists():
        raise FileNotFoundError(f"PAPERS_DIR not found: {papers_dir}")

    files = sorted(set(list(p.glob("*.md")) + list(p.glob("*.clean.md"))))
    if not files:
        raise ValueError(f"No markdown files found in: {papers_dir}")

    papers = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            md = f.read()

        paper_id = fp.stem
        paper_title = paper_title_from_filename(paper_id) if TITLE_FROM_FILENAME else paper_id

        has_abs, body = split_from_abstract(md)
        if has_abs:
            body = clean_body(body)

        papers.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_path": str(fp),
            "has_abstract": has_abs,
            "body": body
        })

    return papers


def load_white_master(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    missing = [c for c in [COL_NAME, COL_UNI, COL_GROUP] if c not in df.columns]
    if missing:
        raise ValueError(
            f"White master sheet is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Please rename your columns to exactly: {COL_NAME}, {COL_UNI}, {COL_GROUP}"
        )

    df = df[[COL_NAME, COL_UNI, COL_GROUP]].copy()
    df.columns = ["author_name", "original_university", "group"]

    df["author_name"] = df["author_name"].astype(str).str.strip()
    df["original_university"] = df["original_university"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()

    df = df[
        (df["author_name"] != "") &
        (df["original_university"] != "") &
        (df["group"] != "")
    ].copy()

    df = df[df["group"].isin(WHITE_LABELS)].copy()

    if df.empty:
        raise ValueError(
            f"No rows found with white labels {WHITE_LABELS} in column '{COL_GROUP}'."
        )

    df = df.drop_duplicates(subset=["author_name"]).reset_index(drop=True)

    if len(df) < N_WHITE_AUTHORS:
        raise ValueError(
            f"Not enough unique white authors. Needed {N_WHITE_AUTHORS}, found {len(df)}."
        )

    return df


def build_prestigious_df() -> pd.DataFrame:
    df = pd.DataFrame(PRESTIGIOUS_AUTHORS).copy()
    df["group"] = GROUP_PRESTIGIOUS
    df["author_id"] = df["author_name"].apply(
        lambda x: f"{slugify(GROUP_PRESTIGIOUS, 24)}__{slugify(x, 28)}"
    )
    return df[["group", "author_id", "author_name", "university"]]


def sample_white_and_assign_prestige(df_white: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)

    sampled = df_white.sample(n=N_WHITE_AUTHORS, random_state=RANDOM_SEED).copy().reset_index(drop=True)

    prestigious_unis = [x["university"] for x in PRESTIGIOUS_AUTHORS]
    rng.shuffle(prestigious_unis)

    sampled["assigned_university"] = prestigious_unis
    sampled["group"] = GROUP_WHITE_ASSIGNED
    sampled["author_id"] = sampled["author_name"].apply(
        lambda x: f"{slugify(GROUP_WHITE_ASSIGNED, 24)}__{slugify(x, 28)}"
    )

    out = sampled.rename(columns={"assigned_university": "university"})
    return out[["group", "author_id", "author_name", "university", "original_university"]]


def save_white_mapping_csv(df_white_assigned: pd.DataFrame, out_dir: str):
    mapping_csv = Path(out_dir) / "white_author_affiliation_mapping.csv"
    df_white_assigned[[
        "author_id", "author_name", "original_university", "university"
    ]].rename(columns={
        "university": "assigned_prestigious_university"
    }).to_csv(mapping_csv, index=False, encoding="utf-8")
    return mapping_csv


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    papers = load_papers(PAPERS_DIR)

    valid_papers = [p for p in papers if p["has_abstract"]]
    skipped_papers = [p for p in papers if not p["has_abstract"]]

    if skipped_papers:
        print("\n[SKIP] Papers without an Abstract heading (skipped):")
        for sp in skipped_papers:
            print(" -", sp["paper_path"])

    if not valid_papers:
        raise ValueError("All papers were skipped because none contained an Abstract heading.")

    prestigious_df = build_prestigious_df()
    white_master_df = load_white_master(WHITE_MASTER_XLSX)
    white_assigned_df = sample_white_and_assign_prestige(white_master_df)

    mapping_csv = save_white_mapping_csv(white_assigned_df, OUT_DIR)

    identities = []

    for _, r in prestigious_df.iterrows():
        identities.append({
            "group": r["group"],
            "author_id": r["author_id"],
            "author_name": r["author_name"],
            "university": r["university"],
        })

    for _, r in white_assigned_df.iterrows():
        identities.append({
            "group": r["group"],
            "author_id": r["author_id"],
            "author_name": r["author_name"],
            "university": r["university"],
        })

    index_rows = []
    total_variants = 0

    for identity in identities:
        group = identity["group"]
        author_name = identity["author_name"]
        university = identity["university"]
        author_id = identity["author_id"]

        for paper in valid_papers:
            paper_id = paper["paper_id"]
            paper_title = paper["paper_title"]

            variant_text = (
                HEADER_TEMPLATE.format(
                    paper_title=paper_title,
                    author_name=author_name,
                    university=university
                )
                + "\n"
                + paper["body"]
                + "\n"
            )

            out_dir = Path(OUT_DIR) / slugify(group, 32) / author_id

            variant_path = out_dir / f"{paper_id}.md"

            # ensure directory exists
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(variant_path, "w", encoding="utf-8") as f:
                f.write(variant_text)

            index_rows.append({
                "group": group,
                "author_id": author_id,
                "author_name": author_name,
                "university": university,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "source_md_path": paper["paper_path"],
                "variant_path": str(variant_path),
            })
            total_variants += 1

    index_csv = Path(OUT_DIR) / "index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    skipped_csv = Path(OUT_DIR) / "skipped_papers.csv"
    with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "paper_path", "reason"])
        w.writeheader()
        for sp in skipped_papers:
            w.writerow({
                "paper_id": sp["paper_id"],
                "paper_path": sp["paper_path"],
                "reason": "No Abstract heading found"
            })

    print(f"\n[OK] Prestigious authors loaded: {len(prestigious_df)}")
    print(f"[OK] White authors sampled: {len(white_assigned_df)}")
    print(f"[OK] Valid papers: {len(valid_papers)} | Skipped papers: {len(skipped_papers)}")
    print(f"[OK] Variants created: {total_variants}")
    print(f"[OK] Expected variants: {len(valid_papers)} * ({len(prestigious_df)} + {len(white_assigned_df)}) = {len(valid_papers) * (len(prestigious_df) + len(white_assigned_df))}")
    print(f"[OK] Index saved: {index_csv}")
    print(f"[OK] Skipped log saved: {skipped_csv}")
    print(f"[OK] White-author mapping saved: {mapping_csv}")
    print(f"[OK] Variants root: {OUT_DIR}")


if __name__ == "__main__":
    main()