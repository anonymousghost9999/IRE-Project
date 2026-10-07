"""
Corpus Ingestion Script: Fetches scientific paper abstracts from arXiv API across 4 distinct categories:
- cs.IR (Information Retrieval & Recommenders)
- cs.CL (Computation & Language / NLP)
- cs.CV (Computer Vision)
- q-bio.QM (Quantitative Biology / Bio-Informatics)

Saves clean records to data/corpus.jsonl.
"""

import os
import time
import json
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Set

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

CATEGORIES = {
    "cs.IR": 1000,
    "cs.CL": 1000,
    "cs.CV": 1000,
    "q-bio.QM": 1000,
}

BATCH_SIZE = 500
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus.jsonl")

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())

def fetch_arxiv_batch(category: str, start: int, max_results: int) -> List[Dict]:
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(url, headers={"User-Agent": "IRE-Project/1.0 (academic research)"})
    
    papers = []
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            root = ET.fromstring(data)
            entries = root.findall(f"{ATOM_NS}entry")
            
            for entry in entries:
                # Extract arXiv ID
                id_elem = entry.find(f"{ATOM_NS}id")
                raw_id = id_elem.text.strip() if id_elem is not None else ""
                paper_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                
                # Title & Abstract
                title_elem = entry.find(f"{ATOM_NS}title")
                title = clean_text(title_elem.text) if title_elem is not None else ""
                
                summary_elem = entry.find(f"{ATOM_NS}summary")
                abstract = clean_text(summary_elem.text) if summary_elem is not None else ""
                
                # Published Date
                pub_elem = entry.find(f"{ATOM_NS}published")
                published = pub_elem.text[:4] if pub_elem is not None and pub_elem.text else "2024"
                
                # Authors
                author_elems = entry.findall(f"{ATOM_NS}author")
                authors = []
                for a in author_elems:
                    name_elem = a.find(f"{ATOM_NS}name")
                    if name_elem is not None and name_elem.text:
                        authors.append(clean_text(name_elem.text))
                
                # Categories
                all_cats = []
                for cat in entry.findall(f"{ATOM_NS}category"):
                    term = cat.attrib.get("term")
                    if term:
                        all_cats.append(term)
                
                prim_cat = entry.find(f"{ARXIV_NS}primary_category")
                primary_cat = prim_cat.attrib.get("term", category) if prim_cat is not None else category
                
                if title and abstract and len(abstract) > 50:
                    papers.append({
                        "paper_id": paper_id,
                        "title": title,
                        "abstract": abstract,
                        "primary_category": primary_cat,
                        "target_domain": category,
                        "all_categories": all_cats,
                        "year": int(published) if published.isdigit() else 2024,
                        "authors": authors[:5]  # first 5 authors
                    })
    except Exception as e:
        print(f"Error fetching {category} (start={start}): {e}")
    
    return papers

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_papers = []
    seen_ids: Set[str] = set()

    print("=" * 70)
    print("STARTING ARXIV DATA INGESTION (PHASE 4)")
    print("=" * 70)

    for cat, total_needed in CATEGORIES.items():
        print(f"\n[+] Fetching category: {cat} (Target: {total_needed} papers)...")
        fetched_for_cat = 0
        start = 0
        
        while fetched_for_cat < total_needed:
            batch_count = min(BATCH_SIZE, total_needed - fetched_for_cat)
            print(f"    --> Requesting {batch_count} papers from offset {start}...")
            batch = fetch_arxiv_batch(cat, start=start, max_results=batch_count)
            
            if not batch:
                print("    [!] Empty batch returned or network error. Pausing and continuing...")
                break
                
            new_papers = 0
            for p in batch:
                if p["paper_id"] not in seen_ids:
                    seen_ids.add(p["paper_id"])
                    all_papers.append(p)
                    new_papers += 1
            
            fetched_for_cat += len(batch)
            start += len(batch)
            print(f"    --> Retrieved {len(batch)} papers ({new_papers} unique). Total collected so far: {len(all_papers)}")
            
            # Rate-limiting compliance with arXiv API guidelines
            time.sleep(3.5)

    print("\n" + "=" * 70)
    print(f"INGESTION COMPLETE: Writing {len(all_papers)} papers to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Successfully saved to {OUTPUT_FILE}")
    print("\nCorpus Distribution by Target Domain:")
    domain_counts = {}
    for p in all_papers:
        d = p["target_domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for d, c in domain_counts.items():
        print(f"  • {d:<12}: {c} papers")
    print("=" * 70)

if __name__ == "__main__":
    main()
