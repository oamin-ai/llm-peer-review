import os
import re
import csv
from pathlib import Path
import pandas as pd


# ---------------- CONFIG ----------------
# Paths are set relative to project structure.
# Modify only if your folder structure is different.

BASE_DIR = Path(__file__).resolve().parent.parent

PAPERS_DIR = "data/processed_papers"
AUTHORS_XLSX = "data/metadata/authors.xlsx"
OUT_DIR = "experiments/ethnicity-bias/variants"

# If you only want some groups, set e.g. {"Germany", "France"} else None
ONLY_GROUPS = None

# You requested: remove everything before Abstract, then rewrite title + new author info
TITLE_FROM_FILENAME = True

# Expected columns in author sheet (you can rename columns in Excel to match these exactly)
COL_NAME = "Name"
COL_UNI = "University"
COL_GROUP = "Ethnicity"   # if your file uses "Germany/France" column named differently, rename to Ethnicity

# Output author block format: ONLY Name + University (no extra fields)
HEADER_TEMPLATE = """# {paper_title}

Author: {author_name}
University: {university}

"""

# Regex: detect Abstract heading in Markdown
ABSTRACT_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*abstract\s*$")
INTRO_HEADING_RE = re.compile(r"(?im)^\s*#{1,6}\s*introduction\s*$")
NEXT_HEADING_LINE_RE = re.compile(r"(?m)^\s*#{1,6}\s+")


# --- NEW: GitHub link line detection (remove the WHOLE line) ---
GITHUB_LINE_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)?"
    r"(?:github\.com|raw\.githubusercontent\.com|gist\.github\.com|github\.io)\b"
)

# --- NEW: Acknowledgment section heading detection ---
ACK_HEADING_RE = re.compile(r"(?im)^\s*(#{1,6})\s*acknowledg(e)?ments?\s*$")
# ----------------------------------------


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
    # handle "*.clean.md" -> stem is "Title_Here.clean"
    if stem.lower().endswith(".clean"):
        stem = stem[:-6]  # remove ".clean"

    title = stem.replace("_", " ").strip()
    title = re.sub(r"\s{2,}", " ", title)  # collapse multiple spaces
    return title



def remove_from_github_to_next_heading(md_text: str) -> str:
    """
    Only looks BEFORE the Introduction heading.
    If 'github' appears, remove from the FIRST 'github' occurrence
    up to (but not including) the NEXT markdown heading line (starts with #).
    If no next heading exists, remove from github to end of the pre-Introduction text.
    """

    m_intro = INTRO_HEADING_RE.search(md_text)
    if m_intro:
        pre = md_text[:m_intro.start()]
        post = md_text[m_intro.start():]
    else:
        pre, post = md_text, ""

    # Find first occurrence of 'github' (case-insensitive) in pre
    m_gh = re.search(r"(?i)\bgithub\b", pre)
    if not m_gh:
        return (pre + post).strip()

    start = m_gh.start()

    # Find next heading line AFTER the github occurrence
    m_next = NEXT_HEADING_LINE_RE.search(pre, pos=start + 1)

    if m_next:
        end = m_next.start()
        pre = pre[:start] + pre[end:]
    else:
        # No heading after github in pre: remove from github to end of pre
        pre = pre[:start]

    # Cleanup spacing a bit
    pre = re.sub(r"\n{3,}", "\n\n", pre)
    pre = re.sub(r"[ \t]{2,}", " ", pre)

    return (pre + post).strip()




def strip_section_by_heading(md_text: str, heading_re: re.Pattern) -> str:
    """
    Removes a Markdown section identified by a heading regex (e.g., Acknowledgements),
    from the heading line until the next heading of SAME or HIGHER level, or end of doc.

    Works for headings like:
      ## Acknowledgements
      ### Acknowledgment
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

        # Found the section start
        level = len(m.group(1))  # number of # in the heading
        i += 1

        # Skip until next heading of level <= current level
        while i < len(lines):
            nxt = lines[i]
            hm = re.match(r"^\s*(#{1,6})\s+.+\S\s*$", nxt)
            if hm:
                nxt_level = len(hm.group(1))
                if nxt_level <= level:
                    break
            i += 1

        # do NOT append the skipped block
        continue

    return "\n".join(out)


def clean_body(md_body: str) -> str:
    md_body = remove_from_github_to_next_heading(md_body)
    md_body = strip_section_by_heading(md_body, ACK_HEADING_RE)
    md_body = re.sub(r"\n{3,}", "\n\n", md_body).strip()
    return md_body



def load_authors(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    # Enforce exact columns (no guessing, no adding extra)
    missing = [c for c in [COL_NAME, COL_UNI, COL_GROUP] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Author sheet is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Please rename your columns to exactly: {COL_NAME}, {COL_UNI}, {COL_GROUP}"
        )

    df = df[[COL_NAME, COL_UNI, COL_GROUP]].copy()
    df.columns = ["author_name", "university", "group"]

    # Clean and drop incomplete rows (do not invent anything)
    df["author_name"] = df["author_name"].astype(str).str.strip()
    df["university"] = df["university"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()

    df = df[(df["author_name"] != "") & (df["university"] != "") & (df["group"] != "")]
    if ONLY_GROUPS is not None:
        df = df[df["group"].isin(ONLY_GROUPS)]

    # Stable ID
    df["author_id"] = df.apply(
        lambda r: f"{slugify(r['group'], 24)}__{slugify(r['author_name'], 28)}",
        axis=1
    )

    return df.reset_index(drop=True)

def split_from_abstract(md_text: str):
    """
    Returns (has_abstract: bool, abstract_and_beyond: str).
    Finds the first Markdown heading that is 'Abstract' (case-insensitive).
    If not found, returns (False, "").
    """
    m = ABSTRACT_HEADING_RE.search(md_text)
    if not m:
        return False, ""
    return True, md_text[m.start():].strip()

def load_papers(papers_dir: str):
    p = Path(papers_dir)
    if not p.exists():
        raise FileNotFoundError(f"PAPERS_DIR not found: {papers_dir}")

    # accept .md and .clean.md
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
            body = clean_body(body)  # ✅ NEW cleaning on body only

        papers.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_path": str(fp),
            "has_abstract": has_abs,
            "body": body
        })

    return papers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    authors = load_authors(AUTHORS_XLSX)
    papers = load_papers(PAPERS_DIR)

    # Filter papers: skip those without Abstract
    valid_papers = [p for p in papers if p["has_abstract"]]
    skipped_papers = [p for p in papers if not p["has_abstract"]]

    if skipped_papers:
        print("\n[SKIP] Papers without an Abstract heading (skipped):")
        for sp in skipped_papers:
            print(" -", sp["paper_path"])

    if not valid_papers:
        raise ValueError("All papers were skipped because none contained an Abstract heading.")

    index_rows = []
    total_variants = 0

    # Build variants
    for _, a in authors.iterrows():
        group = a["group"]
        author_name = a["author_name"]
        university = a["university"]
        author_id = a["author_id"]

        for paper in valid_papers:
            paper_id = paper["paper_id"]
            paper_title = paper["paper_title"]

            # Your requirement: rewrite title + author info, then keep from Abstract onward
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
            out_dir.mkdir(parents=True, exist_ok=True)

            variant_path = out_dir / f"{paper_id}.md"
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

    # Save index.csv
    index_csv = Path(OUT_DIR) / "index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    # Save skipped papers log
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

    print(f"\n[OK] Authors loaded: {len(authors)}")
    print(f"[OK] Valid papers: {len(valid_papers)} | Skipped papers: {len(skipped_papers)}")
    print(f"[OK] Variants created: {total_variants}")
    print(f"[OK] Index saved: {index_csv}")
    print(f"[OK] Skipped log saved: {skipped_csv}")
    print(f"[OK] Variants root: {OUT_DIR}")


if __name__ == "__main__":
    main()
