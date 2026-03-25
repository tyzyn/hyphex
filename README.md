# 🍄 Hyphex

**Agent-authored, citation-grounded knowledge graphs from unstructured documents.**

Hyphex lets AI agents read your documents and produce interlinked wiki pages using Hyphex-flavoured markdown — standard markdown extended with typed entity links and inline citations. Like mycelial networks connecting a forest floor, Hyphex grows a living knowledge graph from your documents.

> ⚠️ **Early development** — this project is under active construction.

## The idea

Most RAG systems chunk documents and embed them. This works for simple Q&A but loses the relationships between concepts and can't trace claims back to sources.

Hyphex takes a different approach: agents act as note-takers, reading source documents and producing focused wiki pages with structured links and grounded citations. The knowledge graph emerges from the notes.

```markdown
---
type: film
director: "[[Christopher Nolan]]"
genre: "[[Science Fiction]]"
year: 2014
---

# Interstellar

[[Christopher Nolan>directed_by]] crafted a visually stunning
exploration of [[General Relativity>explores_theme]] and
humanity's survival instinct. {{1}}

The film's depiction of a black hole was developed with
physicist [[Kip Thorne>consulted_by]] and led to a published
academic paper on gravitational lensing. {{2}}
```