import json
from collections import Counter
from paperfeed.data.db import DatabaseManager
from paperfeed.data.cohort import load_candidate_authors, clean_id

db = DatabaseManager()
authors = load_candidate_authors('data/candidate_authors.jsonl')

author_fields = Counter()
author_subfields = Counter()
author_topics = Counter()

with db.get_connection() as conn:
    for a in authors:
        # Check topics of target paper B
        for t in a.target_paper_b_topics:
            author_topics[t] += 1
            
        pids = [clean_id(p) for p in a.authored_paper_ids_T]
        full_pids = [f"https://openalex.org/{p}" for p in pids]
        all_ids = pids + full_pids
        if not all_ids:
            continue
            
        placeholders = ','.join(['?'] * len(all_ids))
        rows = conn.execute(
            f"""
            SELECT field_id, subfield_id, COUNT(*) as cnt 
            FROM papers 
            WHERE paper_id IN ({placeholders}) AND field_id IS NOT NULL 
            GROUP BY field_id, subfield_id 
            ORDER BY cnt DESC
            """,
            all_ids
        ).fetchall()
        
        if rows:
            top_f = rows[0][0]
            top_sf = rows[0][1]
            if top_f:
                author_fields[top_f] += 1
            if top_sf:
                author_subfields[top_sf] += 1

print("Total Authors Analyzed:", len(authors))
print("Distinct Primary Fields among Authors:", len(author_fields))
print("Distinct Primary Subfields among Authors:", len(author_subfields))

print("\n--- AUTHOR COUNT BY PRIMARY FIELD ---")
for f, c in author_fields.most_common(12):
    f_clean = f.replace("https://openalex.org/fields/", "Field ")
    print(f"{f_clean:12}: {c:4d} authors ({c/1000*100:5.1f}%)")

print("\n--- AUTHOR COUNT BY PRIMARY SUBFIELD ---")
for sf, c in author_subfields.most_common(15):
    sf_clean = sf.replace("https://openalex.org/subfields/", "Subfield ")
    print(f"{sf_clean:15}: {c:4d} authors ({c/1000*100:5.1f}%)")

print("\n--- TOP 10 TARGET RESEARCH TOPICS (Paper B) ---")
for t, c in author_topics.most_common(10):
    print(f"{t:45}: {c:3d} authors")

