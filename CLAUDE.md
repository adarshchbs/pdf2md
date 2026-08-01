# pdf2md project goal

Build a **structure-aware, citation-ready PDF document engine** for high-quality RAG and agentic systems—not merely a plain-text extractor or wrapper around another parser.

Canonical output should preserve document meaning and relationships: sections, paragraphs, sentences, lines, lists, tables, figures, captions, footnotes, page furniture, cross-page continuations, and reading order. Markdown is only one rendered view.

Support coherent hierarchical chunking and stable citations at paragraph, sentence, line, table, row, cell, and labeled-column levels. Citations must trace to source content, page, and geometry. Keep content connected to context needed to understand it, such as section headings, table headers, captions, and footnotes.

Preserve pdf2md’s advantages:

- visual reading order rather than raw PDF object order;
- coherent semantic structure;
- body and page-furniture separation;
- table, figure, and citation relationships;
- explicit, reversible normalization;
- structured output for downstream systems.

Use native extraction as a **content safety floor**, not the desired output. Prefer local fallback for uncertain regions over disabling semantics for a whole document.

Benchmarks are diagnostics, not the product objective. Never tune rules to public examples or accept a change only because one score improves. Every meaningful change must preserve both:

1. **Fidelity:** no foreign content, unexplained loss or duplication, ungrounded rewriting, or broken provenance.
2. **Utility:** no unjustified regression in structure, chunk coherence, reading order, citations, retrieval, or agent reasoning.

Evaluate source grounding, chunk quality, citation correctness, structural integrity, retrieval/agent performance, semantic coverage, fallback rate, and operational cost.

Target: meet a strict content/provenance safety floor while producing more coherent, citation-addressable structure than generic PDF parsers.
