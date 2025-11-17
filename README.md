                        ┌───────────────────────────────┐
                        │       Local PDFs (Your PC)     │
                        │  • naturopathy books           │
                        │  • remedy guides               │
                        └──────────────┬────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────────┐
                     │   Colab Ingestion Pipeline            │
                     │  (process_pdfs.py)                    │
                     │                                       │
                     │  1. Read PDFs                         │
                     │  2. Chunk text (tiktoken)             │
                     │  3. Send each chunk → HF Space        │
                     │     → get 384-dim embedding           │
                     │  4. Insert into Supabase.documents    │
                     │     (chunk, embedding, source)        │
                     └───────────────────────┬───────────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────────┐
                             │       Supabase Database          │
                             │  Table: documents                │
                             │  --------------------------------│
                             │  • id                            │
                             │  • chunk                         │
                             │  • source                        │
                             │  • embedding vector(384)         │
                             │  • chunk_hash (dedupe)           │
                             └──────────────────┬───────────────┘
                                                │
                                                ▼
                     ┌────────────────────────────────────────────────┐
                     │ Supabase RPC: match_documents_v2               │
                     │                                                │
                     │ Input: query_embedding (vector 384)            │
                     │ Logic: similarity search using <=> operator    │
                     │ Output: top N matching chunks                  │
                     └───────────────────┬────────────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │            Render API (FastAPI)            │
                   │--------------------------------------------│
                   │ /fetch_naturopathy_results                 │
                   │                                            │
                   │ 1. Validate auth                           │
                   │ 2. Send query → HF Space embedder          │
                   │ 3. Get embedding                           │
                   │ 4. Call Supabase RPC                       │
                   │ 5. Get matched chunks                      │
                   │ 6. Summarize using GPT OpenAI              │
                   │ 7. Return final natural-language answer    │
                   └────────────────────┬────────────────────────┘
                                        │
                                        ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Squarespace Website (Password-Protected Access)          │
         │----------------------------------------------------------│
         │ • HTML/CSS Chat UI                                       │
         │ • JavaScript fetch() → Render API                        │
         │ • Display final summary to user                          │
         └──────────────────────────────────────────────────────────┘
