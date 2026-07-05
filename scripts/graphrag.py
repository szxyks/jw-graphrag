"""GraphRAG extraction — entity & relationship mining using the local LLM.

Implements the Microsoft GraphRAG pattern:
  1. For each chunk, prompt the LLM to extract entities + relationships
  2. Build a knowledge graph (entities = nodes, relationships = edges)
  3. Run Leiden community detection on the graph
  4. For each community, ask LLM to generate a summary
  5. (Optional) Hierarchical merging of communities

This module is invoked AFTER chunks are inserted into the DB.
"""
import os
import re
import json
import time
from typing import List, Dict, Set, Tuple
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ----- Entity extraction prompt (adapted from Microsoft GraphRAG) -----

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from religious literature.

Given the text below, identify all entities (people, places, concepts, organizations, doctrines, books, events) and the relationships between them.

TEXT:
{text}

Return JSON with this exact structure:
{{
  "entities": [
    {{"name": "Jehovah", "type": "deity", "description": "The supreme God worshipped by Jehovah's Witnesses"}},
    {{"name": "Jesus Christ", "type": "person", "description": "Son of God, Messiah"}}
  ],
  "relationships": [
    {{"source": "Jehovah", "target": "Jesus Christ", "type": "father_of", "description": "Jehovah is the Father of Jesus"}}
  ]
}}

Rules:
- Use lowercase canonical names (e.g. "jehovah", "jesus christ", "kingdom of god", "watchtower society").
- Be specific: "kingdom of god" not just "kingdom".
- Only include relationships where BOTH entities appear in the text.
- Limit to top 8 entities per chunk (skip trivial ones).
- Limit to top 10 relationships per chunk.
- Type options: person, place, deity, concept, organization, doctrine, book, event, object.
- Return ONLY the JSON, no commentary.
"""


SUMMARIZE_CHUNK_PROMPT = """Summarize the following text in 1-2 sentences. Capture the main point only.

Text:
{text}

Summary:"""


COMMUNITY_SUMMARY_PROMPT = """You are summarizing a community of related concepts extracted from Jehovah's Witnesses publications.

The following entities form a connected community:
{entities}

Sample text from chunks that mention these entities:
{sample_text}

Write a comprehensive 3-5 sentence summary of what this community represents. Focus on:
- The main theme or doctrine
- How these entities relate to each other
- Why they appear together in JW publications

Summary:"""


def call_llm(prompt: str, model: str = None, temperature: float = 0.1) -> str:
    """Non-streaming LLM call for extraction (low temperature for determinism)."""
    model = model or LLM_MODEL
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def call_embed(text: str, model: str = None) -> List[float]:
    model = model or EMBED_MODEL
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": model, "prompt": text[:8000]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def extract_entities_from_chunk(text: str) -> Dict:
    """Extract entities + relationships from a chunk using the LLM.
    Returns: {"entities": [...], "relationships": [...]}
    """
    # Truncate very long chunks to fit context
    truncated = text[:4000] if len(text) > 4000 else text
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=truncated)

    try:
        response = call_llm(prompt, temperature=0.1)
        # Extract JSON from response (LLMs sometimes wrap in markdown)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return {"entities": [], "relationships": []}
        parsed = json.loads(json_match.group())
        # Validate
        if not isinstance(parsed, dict):
            return {"entities": [], "relationships": []}
        return {
            "entities": parsed.get("entities", [])[:8],
            "relationships": parsed.get("relationships", [])[:10],
        }
    except (json.JSONDecodeError, requests.RequestException) as e:
        return {"entities": [], "relationships": []}


def summarize_chunk(text: str) -> str:
    """Generate a 1-2 sentence summary of a chunk."""
    try:
        truncated = text[:2000] if len(text) > 2000 else text
        prompt = SUMMARIZE_CHUNK_PROMPT.format(text=truncated)
        return call_llm(prompt, temperature=0.3).strip()
    except Exception:
        return ""


def summarize_community(entities: List[Dict], sample_text: str) -> str:
    """Generate a summary for a community of entities."""
    try:
        ent_str = "\n".join(f"- {e['name']} ({e.get('type','?')}): {e.get('description','')[:100]}"
                            for e in entities[:30])
        prompt = COMMUNITY_SUMMARY_PROMPT.format(entities=ent_str, sample_text=sample_text[:3000])
        return call_llm(prompt, temperature=0.4).strip()
    except Exception:
        return ""


# ----- Community detection (Leiden via networkx) -----

def detect_communities(entities: List[Dict], relationships: List[Dict]) -> Dict[int, List[int]]:
    """Run community detection on the entity graph.

    Returns: {community_id: [entity_index, ...]}
    """
    try:
        import networkx as nx
    except ImportError:
        # Fallback: connected components
        return _connected_components(entities, relationships)

    G = nx.Graph()
    for i, e in enumerate(entities):
        G.add_node(i, name=e["name"], type=e.get("type"))
    for r in relationships:
        src = r.get("source", "").lower()
        tgt = r.get("target", "").lower()
        src_idx = next((i for i, e in enumerate(entities) if e["name"].lower() == src), None)
        tgt_idx = next((i for i, e in enumerate(entities) if e["name"].lower() == tgt), None)
        if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
            if G.has_edge(src_idx, tgt_idx):
                G[src_idx][tgt_idx]["weight"] += 1
            else:
                G.add_edge(src_idx, tgt_idx, weight=1)

    # Try Leiden (python-igraph); fall back to Louvain (networkx)
    try:
        import leidenalg
        import igraph as ig
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(entities))
        edges = [(u, v) for u, v in G.edges()]
        if edges:
            ig_graph.add_edges(edges)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, weights=weights)
        communities = {}
        for i, c in enumerate(partition.membership):
            communities.setdefault(c, []).append(i)
        return communities
    except ImportError:
        pass

    # Louvain via networkx
    try:
        communities_dict = nx.community.louvain_communities(G, weight="weight", seed=42)
        return {i: list(c) for i, c in enumerate(communities_dict)}
    except Exception:
        pass

    # Final fallback: connected components
    return _connected_components(entities, relationships, G)


def _connected_components(entities, relationships, G=None) -> Dict[int, List[int]]:
    """Fallback: use connected components as communities."""
    if G is None:
        import networkx as nx
        G = nx.Graph()
        for i, e in enumerate(entities):
            G.add_node(i)
        for r in relationships:
            src = r.get("source", "").lower()
            tgt = r.get("target", "").lower()
            src_idx = next((i for i, e in enumerate(entities) if e["name"].lower() == src), None)
            tgt_idx = next((i for i, e in enumerate(entities) if e["name"].lower() == tgt), None)
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                G.add_edge(src_idx, tgt_idx)
    communities = {}
    for i, comp in enumerate(nx.connected_components(G)):
        communities[i] = list(comp)
    return communities


# ----- Canonicalization -----

def canonicalize_entity_name(name: str) -> str:
    """Normalize entity name to canonical form."""
    name = name.strip().lower()
    # Map common aliases
    aliases = {
        "jehovah god": "jehovah",
        "god": "jehovah",
        "lord": "jehovah",
        "jesus": "jesus christ",
        "christ": "jesus christ",
        "christ jesus": "jesus christ",
        "holy spirit": "holy spirit",
        "god's spirit": "holy spirit",
        "kingdom": "kingdom of god",
        "god's kingdom": "kingdom of god",
        "the kingdom": "kingdom of god",
        "the watchtower": "watchtower",
        "watchtower society": "watchtower",
        "jehovah's witnesses": "jehovah's witnesses",
        "jws": "jehovah's witnesses",
        "bible": "bible",
        "the bible": "bible",
        "scriptures": "bible",
    }
    return aliases.get(name, name)
