# Six Core Retrieval-Augmented Generation (RAG) Architectures

## Document Information
- **System:** IncidentLens Outage Intelligence System
- **Language Standard:** ASD-STE100 (Simplified Technical English)
- **Document Type:** Technical Specification and Training Guide

---

## 1. Introduction

This document describes the six core RAG architectures in the IncidentLens project.
Each architecture operates as a layer in one unified engine.
The system uses these architectures to analyze system outage reports.

The table below lists the six architectures and their system layers:

| Number | Architecture Name | System Layer | Primary Function |
|---|---|---|---|
| 1 | Contextual Retrieval | Ingestion Layer | Adds incident background to each chunk |
| 2 | Late Chunking | Ingestion Layer | Preserves attention across the full timeline |
| 3 | GraphRAG | Storage Layer | Finds multi-hop failure links between services |
| 4 | Multimodal RAG | Storage Layer | Reads charts and diagrams from PDF reports |
| 5 | Self-Correcting Agentic RAG | Execution Layer | Tests document relevance and corrects errors |
| 6 | Long Context vs. RAG Router | Strategy Layer | Selects the best retrieval path for the query |

---

## 2. Architecture 1: Contextual Retrieval

### 2.1 What It Is
Contextual Retrieval is a document processing method.
This method adds a short summary header to every text chunk before the system creates vector embeddings.
The summary header contains the incident name, date, affected service, and primary cause.

### 2.2 Why We Use It
Standard document splitters create small text chunks.
These chunks frequently lose their original context.
A chunk can contain pronouns such as "it" or "the server".
The vector search engine cannot match search queries to ambiguous text.
When we prepend the summary header, each chunk becomes independent.
The search engine can find the correct chunk without ambiguity.

### 2.3 How It Operates in This Project
1. The ingestion pipeline reads a post-mortem document.
2. A language model generates a 50-word situation header for the incident.
3. The system adds this header to the start of each chunk.
4. The BGE-M3 model creates embeddings for the augmented chunks.
5. ChromaDB stores the vectors and the augmented text.

```
Example:

Original Chunk:
"The thread pool was empty. New connections failed immediately."

Contextual Chunk:
"[CONTEXT: AWS US-East-1 Kinesis Outage, Nov 2020. Front-end OS thread exhaustion.]
The thread pool was empty. New connections failed immediately."
```

### 2.4 Technical Rules
- Keep the situation header below 50 words.
- Include the system name, outage date, and root cause in the header.
- Store the original raw text in metadata for clean response generation.

---

## 3. Architecture 2: Late Chunking

### 2.1 What It Is
Late Chunking is an embedding technique.
The system inputs the complete document into an embedding model before it cuts the text into chunks.
The model calculates token embeddings with attention across the entire document.
After the model processes the document, the system groups token vectors into chunk boundaries.

### 2.2 Why We Use It
Standard chunking divides text before embedding.
When the system splits text early, tokens in chunk A cannot attend to tokens in chunk B.
Outage post-mortems contain long timelines.
An initial configuration error at 14:00 hours can cause a database crash at 20:00 hours.
Standard chunking breaks the relationship between these two events.
Late Chunking keeps long-range attention active across the complete outage timeline.

### 2.3 How It Operates in This Project
1. The system loads the complete incident timeline text.
2. The BGE-M3 embedding model processes all document tokens in one pass.
3. Every token generates a vector that contains context from all other tokens.
4. The system applies chunk boundary limits to the token sequence.
5. Mean-pooling combines the token vectors for each chunk.
6. The system saves the pooled chunk vectors to ChromaDB.

```
Standard Chunking:
Document -> Split Into Chunks -> Embed Chunks (Cross-chunk attention is lost)

Late Chunking:
Document -> Embed Full Text -> Group Token Vectors (Cross-chunk attention is kept)
```

### 2.4 Technical Rules
- Do not exceed the maximum context window of the embedding model.
- Use mean-pooling on token spans to make chunk embeddings.
- Record start and end token offsets for every chunk.

---

## 4. Architecture 3: GraphRAG (Multi-Hop Failure Chains)

### 3.1 What It Is
GraphRAG is a retrieval architecture that combines a knowledge graph with vector search.
The knowledge graph stores system components as nodes.
The graph stores dependencies and failure paths as directed edges.

### 3.2 Why We Use It
Standard vector search finds text chunks with similar words.
Vector search cannot follow cascading failure chains across multiple services.
In microservice architectures, one failure triggers downstream failures.
For example, a DNS leak causes queue saturation, which causes database exhaustion.
Vector search often retrieves the last error but misses the root trigger.
GraphRAG traverses directed edges to find the true root cause across multiple hops.

### 3.3 How It Operates in This Project
1. An entity extraction model identifies services, errors, and resources in post-mortems.
2. The system creates directed relationship edges between entities.
3. NetworkX manages the graph in memory during development.
4. Neo4j stores large graphs in production.
5. When a user asks for a root cause, the graph engine traces paths backwards from the symptom.
6. The system combines graph paths with text chunks to form the final prompt.

```
Failure Graph Path:
[DNS Route Leak] ---> [Kafka Queue Full] ---> [DB Pool Exhausted] ---> [API Gateway 503]
(Hop 1)                (Hop 2)                 (Hop 3)
```

### 3.4 Technical Rules
- Extract three entity types: Service, FailureMode, and InfrastructureComponent.
- Limit graph traversals to a maximum depth of 4 hops.
- Return both the node path and the source citations in the retrieval output.

---

## 5. Architecture 4: Multimodal RAG

### 4.1 What It Is
Multimodal RAG is an architecture that extracts and retrieves information from text and images.
The system processes visual items such as latency graphs, system topology diagrams, and memory charts.

### 4.2 Why We Use It
Technical post-mortem documents contain important graphics.
Grafana dashboards show exact traffic spikes and resource limits.
Architecture diagrams show network routes and firewalls.
Standard text extraction tools delete or ignore image data.
When the system ignores images, it loses critical diagnostic evidence.
Multimodal RAG converts graphic evidence into searchable information.

### 4.3 How It Operates in This Project
1. The ingestion pipeline extracts image frames from post-mortem PDF files.
2. A vision model (via OpenRouter Free Vision API) analyzes each chart and diagram.
3. The vision model generates a detailed technical description of the image.
4. The system indexes the image description in the hybrid search store.
5. At query time, the retrieval engine retrieves matching text and chart descriptions.
6. The generator uses both data sources to answer technical questions.

```
Processing Flow:
PDF Page Image -> Vision API -> Structured Chart Summary -> Vector Store + Text Prompt
```

### 4.4 Technical Rules
- Extract images at a minimum resolution of 150 DPI.
- The image description must contain axes, units, peak values, and anomalies.
- Store image coordinates and page numbers in the chunk metadata.

---

## 6. Architecture 5: Self-Correcting Agentic RAG

### 5.1 What It Is
Self-Correcting Agentic RAG is an orchestration architecture with a control loop.
The architecture uses an autonomous agent to test retrieval quality, rewrite queries, and remove hallucinations.
The agent operates through a LangGraph state machine.

### 5.2 Why We Use It
Naive RAG executes a single linear sequence: Retrieve, Augment, and Generate.
If the retrieval step returns irrelevant text, Naive RAG produces incorrect answers.
If the language model hallucinates, Naive RAG cannot detect the error.
Production systems need verification and recovery mechanisms.
Agentic RAG evaluates documents before generation.
Agentic RAG evaluates answers after generation.
The loop rewrites queries when retrieved data is insufficient.

### 5.3 How It Operates in This Project
1. **Retrieve:** The system retrieves documents from vector and graph stores.
2. **Grade Relevance:** A grading node inspects each document.
3. **Decide Path:**
   - If documents are relevant: Send data to the generator node.
   - If documents are irrelevant: Rewrite query and search again.
4. **Generate:** The language model generates an incident response.
5. **Check Hallucination:** A test node checks if facts match the retrieved text.
   - If facts do not match: Re-generate the response.
   - If internal data is insufficient: Execute fallback web search.
6. **Output:** Return the verified response with citations.

```
Agentic Loop:
User Query
   |
   v
[Retrieve Chunks]
   |
   v
[Grade Chunks] ---> (Low Quality) ---> [Rewrite Query] ---> (Back to Retrieve)
   |
(High Quality)
   v
[Generate Answer]
   |
   v
[Check Hallucination] ---> (Hallucinated) ---> [Re-Generate Answer]
   |
(Grounded)
   v
Verified Response
```

### 5.4 Technical Rules
- Limit retry loops to a maximum of 3 iterations.
- Mark all ungrounded statements as errors.
- Record every decision node in Langfuse and LangSmith traces.

---

## 7. Architecture 6: Long Context vs. RAG Router

### 6.1 What It Is
The Long Context vs. RAG Router is a decision engine.
The router inspects the incoming user query and selects the most effective execution strategy:
- Strategy A: Targeted RAG pipeline (fast and specific).
- Strategy B: Full-document long-context processing (broad and comprehensive).

### 6.2 Why We Use It
Modern language models support context windows larger than one million tokens.
However, processing one million tokens for every question is slow and expensive.
Targeted RAG is superior for narrow facts, error codes, and configuration parameters.
Targeted RAG costs less and responds faster.
Conversely, targeted RAG struggles with full-incident narrative summaries and global timelines.
Full-context processing is superior for comprehensive analysis of 200-page reports.
The router provides the lowest latency and cost for each query type.

### 6.3 How It Operates in This Project
1. A router node analyzes the user prompt.
2. The router classifies query intent into two categories:
   - **Narrow Fact Query:** Error codes, single timestamps, specific command flags.
   - **Global Synthesis Query:** Full timelines, multi-page summaries, comparative incident reviews.
3. If the query is narrow: The router calls the Hybrid Agentic RAG pipeline.
4. If the query is global: The router sends the complete raw document to the long-context model.
5. The router records execution latency and token cost for comparison.

```
Routing Logic:
User Query -> [Classifier]
                 |
                 +--> (Narrow Root Cause)  -> [Agentic Hybrid RAG] -> Fast Response
                 |
                 +--> (Broad Narrative)    -> [1M Context Window]  -> Complete Summary
```

### 6.4 Technical Rules
- The classifier must complete execution in less than 200 milliseconds.
- Use targeted RAG for documents that exceed 500,000 tokens when the query is specific.
- Use long-context processing when the query requires cross-referencing more than five sections.

---

## 8. Summary Comparison Table

The table below summarizes the characteristics of the six architectures:

| Architecture | Primary Input | Main Advantage | Failure Mode Prevented |
|---|---|---|---|
| **Contextual Retrieval** | Document chunks | Preserves incident context | Orphan pronouns and lost references |
| **Late Chunking** | Full timeline text | Preserves long-range attention | Broken cause-and-effect links |
| **GraphRAG** | Entities and dependencies | Follows cascading failures | Missed multi-hop root causes |
| **Multimodal RAG** | Text and PDF images | Extracts graphic metrics | Loss of chart and diagram data |
| **Agentic RAG** | Queries and retrieved text | Verifies facts in a loop | Hallucinations and irrelevant data |
| **Long Context Router** | Query complexity metadata | Balances cost and speed | High latency on simple questions |

---

## 9. Verification Checklist

Use this checklist to test compliance with project standards:

- [ ] All sentences use direct active voice.
- [ ] Descriptive sentences contain 25 words or fewer.
- [ ] Procedural sentences contain 20 words or fewer.
- [ ] Jargon words without technical necessity are removed.
- [ ] Each architecture section contains: What It Is, Why We Use It, and How It Operates.
