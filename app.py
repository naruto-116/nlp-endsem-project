"""
KG-CiteRAG: Knowledge-Graph-Augmented & Citation-Enforced Retrieval System
Main Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from src.retrieval import HybridRetriever
from src.generator import LegalAnswerGenerator, DummyGenerator
from src.verifier import CitationVerifier
from src.graph_utils import LegalKnowledgeGraph
from src.pdf_processor import PDFProcessor
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="KG-CiteRAG: Legal QA System",
    page_icon="⚖️",
    layout="wide"
)


def rebuild_uploaded_index(system):
    """Rebuild FAISS index from uploaded documents."""
    if not st.session_state.uploaded_docs:
        st.session_state.uploaded_index = None
        st.session_state.uploaded_metadata = []
        return
    
    # Get model
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Collect all chunks
    all_chunks = []
    all_metadata = []
    
    for doc in st.session_state.uploaded_docs:
        for chunk_idx, chunk_data in enumerate(doc['chunks']):
            # Handle both old string format and new dict format
            if isinstance(chunk_data, dict):
                chunk_text = chunk_data['text']
                page_num = chunk_data.get('page_num', 1)
            else:
                chunk_text = chunk_data
                page_num = 1
            
            all_chunks.append(chunk_text)
            all_metadata.append({
                'filename': doc['filename'],
                'chunk_id': chunk_idx,
                'text': chunk_text,
                'page_num': page_num,
                'source_type': 'uploaded_pdf'
            })
    
    # Create embeddings
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    st.session_state.uploaded_index = index
    st.session_state.uploaded_metadata = all_metadata

def add_uploaded_docs_to_graph(system):
    """Add uploaded documents as nodes to the knowledge graph."""
    import networkx as nx
    
    graph = system['graph'].graph
    
    for doc in st.session_state.uploaded_docs:
        filename = doc['filename']
        
        # Add document as node
        if filename not in graph:
            graph.add_node(filename, name=filename, type='uploaded_pdf')
        
        # Add citation edges
        citations = doc.get('citations', [])
        for citation in citations[:20]:  # Limit to first 20 citations
            # Try to find matching node in graph
            citation_clean = citation.strip()
            
            # Search for case nodes that match
            for node in graph.nodes():
                node_name = graph.nodes[node].get('name', '')
                if citation_clean in node_name or node_name in citation_clean:
                    # Add edge from PDF to cited case
                    graph.add_edge(filename, node, relation='cites')
                    break

def search_uploaded_docs(query, system, top_k=5):
    """Search uploaded documents."""
    if st.session_state.uploaded_index is None or st.session_state.uploaded_index.ntotal == 0:
        return []
    
    # Get model
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Encode query
    query_embedding = model.encode([query])[0].astype('float32')
    
    # Search
    distances, indices = st.session_state.uploaded_index.search(
        query_embedding.reshape(1, -1),
        min(top_k, st.session_state.uploaded_index.ntotal)
    )
    
    # Format results with page numbers
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(st.session_state.uploaded_metadata):
            meta = st.session_state.uploaded_metadata[idx]
            page_num = meta.get('page_num', 1)
            filename = meta['filename']
            
            results.append({
                'case_name': f"{filename}-page{page_num}",
                'text': meta['text'],
                'score': float(1 / (1 + dist)),
                'source_type': 'uploaded_pdf',
                'source_filename': filename,
                'page_num': page_num,
                'chunk_id': meta['chunk_id']
            })
    
    return results

@st.cache_resource
def load_system():
    """Load all system components (cached for performance)."""
    try:
        # Check if required files exist
        if not config.GRAPH_PATH.exists():
            st.error(f"❌ Knowledge Graph not found at {config.GRAPH_PATH}")
            st.info("Please run: `python scripts/build_knowledge_graph.py`")
            return None
        
        if not config.VECTOR_INDEX_PATH.exists():
            st.error(f"❌ Vector Index not found at {config.VECTOR_INDEX_PATH}")
            st.info("Please run: `python scripts/build_vector_index.py`")
            return None
        
        # Load components
        with st.spinner("Loading retrieval system..."):
            retriever = HybridRetriever(
                index_path=config.VECTOR_INDEX_PATH,
                metadata_path=config.METADATA_PATH,
                graph_path=config.GRAPH_PATH,
                embedding_model_name=config.EMBEDDING_MODEL
            )
        
        # Load knowledge graph for verification
        graph = LegalKnowledgeGraph.load(config.GRAPH_PATH)
        verifier = CitationVerifier(graph)
        
        # Load generator
        if config.GEMINI_API_KEY:
            generator = LegalAnswerGenerator(
                api_key=config.GEMINI_API_KEY,
                model_name=config.LLM_MODEL,
                provider="gemini"
            )
        elif config.GROQ_API_KEY:
            generator = LegalAnswerGenerator(
                api_key=config.GROQ_API_KEY,
                model_name=config.LLM_MODEL,
                provider="groq"
            )
        else:
            st.warning("⚠️ No API key found. Using dummy generator.")
            st.info("Add your API key to .env file: GEMINI_API_KEY=your_key_here or GROQ_API_KEY=your_key_here")
            generator = DummyGenerator()
        
        return {
            'retriever': retriever,
            'generator': generator,
            'verifier': verifier,
            'graph': graph
        }
        
    except Exception as e:
        st.error(f"Error loading system: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def main():
    """Main application."""
    
    # Title and description
    st.title("⚖️ KG-CiteRAG: Legal Question Answering System")
    st.markdown("""
    A hybrid Knowledge-Graph-Augmented & Citation-Enforced Retrieval system for Indian Supreme Court judgments.
    Ask legal questions or upload documents for verified, citation-backed answers.
    """)
    
    # Initialize session state for uploaded documents
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'uploaded_index' not in st.session_state:
        st.session_state.uploaded_index = None
    if 'uploaded_metadata' not in st.session_state:
        st.session_state.uploaded_metadata = []
    
    # Load system
    system = load_system()
    
    if not system:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider("Number of retrieved documents", 5, 20, 10)
        vector_weight = 0.7
        graph_weight = 0.3
        
        st.markdown("---")
        st.header("📄 Uploaded Documents")
        st.metric("Uploaded", len(st.session_state.uploaded_docs))
        
        if st.session_state.uploaded_docs:
            for i, doc_info in enumerate(st.session_state.uploaded_docs):
                with st.expander(f"📄 {doc_info['filename'][:25]}..."):
                    st.text(f"Chunks: {doc_info['num_chunks']}")
                    st.text(f"Words: {doc_info['total_words']}")
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.uploaded_docs.pop(i)
                        # Rebuild index
                        rebuild_uploaded_index(system)
                        st.rerun()
        
        st.markdown("---")
        st.header("📊 System Stats")
        stats = system['graph'].get_graph_stats()
        st.metric("Total Cases", f"{stats['num_cases']:,}")
        st.metric("Total Citations", f"{stats['num_citations']:,}")
        st.metric("Avg Citations/Case", f"{stats['avg_citations_per_case']:.1f}")
        
        st.markdown("---")
        st.header("🏛️ Landmark Cases")
        landmark_cases = system['graph'].get_landmark_cases(top_n=5)
        for i, (case_id, count) in enumerate(landmark_cases, 1):
            case_name = system['graph'].graph.nodes[case_id].get('name', case_id)
            st.text(f"{i}. {case_name[:40]}... ({count})")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📄 Upload Document", "ℹ️ About"])
    
    with tab1:
        st.header("Ask a Legal Question")
        
        # Example queries
        st.markdown("**Example queries:**")
        examples = [
            "What are the grounds for divorce under Hindu Law?",
            "Explain the right to privacy under Article 21",
            "What are the Supreme Court guidelines on arrest procedures?"
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(examples):
            if cols[i].button(f"Try: {example[:30]}...", key=f"ex{i}"):
                st.session_state.query = example
        
        # Query input
        query = st.text_area(
            "Enter your legal question:",
            value=st.session_state.get('query', ''),
            height=100,
            key="query_input"
        )
        
        if st.button("🔍 Search & Generate Answer", type="primary"):
            if not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving relevant cases..."):
                    # Search uploaded documents first
                    uploaded_results = search_uploaded_docs(query, system, top_k=5)
                    
                    # Hybrid retrieval from main index
                    main_results = system['retriever'].hybrid_search(
                        query=query,
                        top_k=top_k,
                        vector_weight=vector_weight,
                        graph_weight=graph_weight
                    )
                    
                    # Combine results (prioritize uploaded docs)
                    results = uploaded_results + main_results
                    results = results[:top_k]  # Limit to top_k
                    
                    # Show retrieved documents
                    with st.expander("📚 Retrieved Documents", expanded=False):
                        for i, result in enumerate(results, 1):
                            case_name = result.get('case_name', 'Unknown Case')
                            score = result.get('combined_score', result.get('score', 0))
                            text = result.get('text', '')
                            source_type = result.get('source_type', 'ildc')
                            
                            # Format source label with page number for uploaded PDFs
                            if source_type == 'uploaded_pdf':
                                filename = result.get('source_filename', case_name)
                                page_num = result.get('page_num', 1)
                                source_label = f"📄 {filename} (Page {page_num})"
                            else:
                                source_label = "🏛️ ILDC Dataset"
                            
                            st.markdown(f"**{i}. {case_name}**")
                            st.text(f"Score: {score:.3f} | Source: {source_label}")
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.markdown("---")
                
                # Check if we got any results
                if not results:
                    st.warning("⚠️ No relevant documents found for your query. Try rephrasing your question.")
                else:
                    with st.spinner("Generating answer..."):
                        # Get context and generate
                        context = system['retriever'].get_context_for_generation(results)
                        answer_dict = system['generator'].generate_answer(query, context)
                        raw_answer = answer_dict.get('answer', 'Could not generate answer.')
                        
                        # Check for errors/fallback
                        if 'error' in answer_dict:
                            st.warning(f"⚠️ API Issue: {answer_dict['error']}")
                            st.info("💡 Showing fallback response with extracted content from retrieved documents.")
                    
                    with st.spinner("Verifying citations..."):
                        # Verify citations
                        corrected_answer, report = system['verifier'].correct_answer(raw_answer)
                
                    # Display results
                    st.markdown("---")
                    st.header("📝 Answer")
                    st.markdown(corrected_answer)
                    
                    # Verification report
                    st.markdown("---")
                    st.header("✅ Citation Verification Report")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Citations", report.get('total_citations', 0))
                    col2.metric("Verified", report.get('verified', 0), 
                               delta="✓" if report.get('verified', 0) > 0 else None)
                    col3.metric("Not Found", report.get('not_found', 0),
                               delta="⚠️" if report.get('not_found', 0) > 0 else None)
                    
                    # Show details
                    if report.get('details'):
                        with st.expander("📋 Citation Details", expanded=True):
                            for detail in report['details']:
                                status_emoji = "✅" if detail.get('verified', False) else "❌"
                                st.markdown(f"{status_emoji} **{detail.get('citation', 'Unknown')}** - {detail.get('status', 'Unknown')}")
                    
                    # Knowledge Graph Visualization - ONLY Uploaded PDFs
                    st.markdown("---")
                    st.header("🕸️ Knowledge Graph - Uploaded Documents")
                    
                    if st.session_state.uploaded_docs:
                        try:
                            import networkx as nx
                            import matplotlib.pyplot as plt
                            import re
                            
                            # Create separate graph for each PDF
                            for doc_idx, doc in enumerate(st.session_state.uploaded_docs):
                                filename = doc['filename']
                                citations = doc.get('citations', [])
                                
                                st.markdown(f"### 📄 {filename}")
                                
                                # Build individual knowledge graph for this PDF
                                pdf_graph = nx.DiGraph()
                                
                                # Add PDF as central node
                                pdf_graph.add_node(filename, type='document', color='lightgreen')
                                
                                # Extract cited cases and add them as nodes
                                for citation in citations[:15]:  # Limit to 15 for clarity
                                    citation_clean = citation.strip()
                                    if len(citation_clean) > 5:  # Valid citation
                                        # Add citation as node
                                        pdf_graph.add_node(citation_clean, type='cited_case', color='lightblue')
                                        # Add edge with relationship
                                        pdf_graph.add_edge(filename, citation_clean, relation='cites')
                                
                                # Extract key legal terms/articles mentioned
                                chunks = doc.get('chunks', [])
                                if chunks:
                                    # Combine some text for analysis
                                    sample_text = ""
                                    for chunk in chunks[:5]:
                                        if isinstance(chunk, dict):
                                            sample_text += chunk.get('text', '') + " "
                                        else:
                                            sample_text += chunk + " "
                                    
                                    # Extract Article mentions
                                    articles = re.findall(r'Article\s+\d+[A-Z]?', sample_text, re.IGNORECASE)
                                    articles = list(set(articles))[:5]  # Limit to 5 unique articles
                                    
                                    for article in articles:
                                        article_clean = article.strip()
                                        pdf_graph.add_node(article_clean, type='article', color='lightyellow')
                                        pdf_graph.add_edge(filename, article_clean, relation='discusses')
                                    
                                    # Extract High Courts mentioned
                                    courts = re.findall(r'([\w\s]+)\s+High\s+Court', sample_text, re.IGNORECASE)
                                    courts = list(set([c.strip() + ' High Court' for c in courts]))[:3]
                                    
                                    for court in courts:
                                        pdf_graph.add_node(court, type='court', color='lightcoral')
                                        pdf_graph.add_edge(filename, court, relation='mentions')
                            
                                if len(pdf_graph.nodes()) > 1:
                                    # Create visualization
                                    fig, ax = plt.subplots(figsize=(14, 9))
                                    
                                    # Use hierarchical layout for clarity
                                    pos = nx.spring_layout(pdf_graph, k=3, iterations=100, seed=42)
                                    
                                    # Separate nodes by type
                                    doc_nodes = [n for n, d in pdf_graph.nodes(data=True) if d.get('type') == 'document']
                                    case_nodes = [n for n, d in pdf_graph.nodes(data=True) if d.get('type') == 'cited_case']
                                    article_nodes = [n for n, d in pdf_graph.nodes(data=True) if d.get('type') == 'article']
                                    court_nodes = [n for n, d in pdf_graph.nodes(data=True) if d.get('type') == 'court']
                                    
                                    # Draw nodes by type with different colors
                                    if doc_nodes:
                                        nx.draw_networkx_nodes(pdf_graph, pos, nodelist=doc_nodes,
                                                              node_color='#90EE90', node_size=4000, 
                                                              alpha=0.9, ax=ax, label='📄 This Document', node_shape='s')
                                    
                                    if case_nodes:
                                        nx.draw_networkx_nodes(pdf_graph, pos, nodelist=case_nodes,
                                                              node_color='#87CEEB', node_size=2500,
                                                              alpha=0.8, ax=ax, label='⚖️ Cited Cases')
                                    
                                    if article_nodes:
                                        nx.draw_networkx_nodes(pdf_graph, pos, nodelist=article_nodes,
                                                              node_color='#FFFFE0', node_size=2000,
                                                              alpha=0.8, ax=ax, label='📜 Articles')
                                    
                                    if court_nodes:
                                        nx.draw_networkx_nodes(pdf_graph, pos, nodelist=court_nodes,
                                                              node_color='#F08080', node_size=2200,
                                                              alpha=0.8, ax=ax, label='🏛️ Courts')
                                    
                                    # Draw edges with different styles based on relation
                                    cites_edges = [(u, v) for u, v, d in pdf_graph.edges(data=True) if d.get('relation') == 'cites']
                                    discusses_edges = [(u, v) for u, v, d in pdf_graph.edges(data=True) if d.get('relation') == 'discusses']
                                    mentions_edges = [(u, v) for u, v, d in pdf_graph.edges(data=True) if d.get('relation') == 'mentions']
                                    
                                    if cites_edges:
                                        nx.draw_networkx_edges(pdf_graph, pos, edgelist=cites_edges, 
                                                              edge_color='blue', arrows=True, arrowsize=15, 
                                                              ax=ax, width=2, style='solid', alpha=0.7)
                                    
                                    if discusses_edges:
                                        nx.draw_networkx_edges(pdf_graph, pos, edgelist=discusses_edges,
                                                              edge_color='green', arrows=True, arrowsize=15,
                                                              ax=ax, width=2, style='dashed', alpha=0.7)
                                    
                                    if mentions_edges:
                                        nx.draw_networkx_edges(pdf_graph, pos, edgelist=mentions_edges,
                                                              edge_color='red', arrows=True, arrowsize=15,
                                                              ax=ax, width=2, style='dotted', alpha=0.7)
                                    
                                    # Draw edge labels with relationships
                                    edge_labels = nx.get_edge_attributes(pdf_graph, 'relation')
                                    nx.draw_networkx_edge_labels(pdf_graph, pos, edge_labels, 
                                                                font_size=9, font_weight='bold',
                                                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                                                ax=ax)
                                    
                                    # Draw node labels (truncate long names)
                                    labels = {}
                                    for node in pdf_graph.nodes():
                                        if len(str(node)) > 40:
                                            labels[node] = str(node)[:37] + '...'
                                        else:
                                            labels[node] = str(node)
                                    
                                    nx.draw_networkx_labels(pdf_graph, pos, labels, font_size=9, 
                                                           font_weight='bold', ax=ax)
                                    
                                    # Add legend with clear descriptions
                                    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
                                    
                                    ax.set_title(f"📊 Knowledge Graph: {filename}", 
                                               fontsize=16, fontweight='bold', pad=20)
                                    ax.axis('off')
                                    
                                    # Add explanation
                                    fig.text(0.5, 0.02, 
                                            'Solid lines = cites | Dashed lines = discusses | Dotted lines = mentions',
                                            ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    # Show graph statistics for this PDF
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Total Nodes", len(pdf_graph.nodes()))
                                    col2.metric("Total Relationships", len(pdf_graph.edges()))
                                    col3.metric("Citations Found", len(citations))
                                else:
                                    st.info(f"📊 No relationships found in {filename}.")
                                
                                # Add separator between PDFs
                                if doc_idx < len(st.session_state.uploaded_docs) - 1:
                                    st.markdown("---")
                        except Exception as e:
                            st.error(f"Could not generate graph visualization: {str(e)}")
                            st.code(traceback.format_exc())
                    else:
                        st.info("📊 No uploaded documents yet. Upload a PDF in the 'Upload Document' tab to see its knowledge graph.")
    
    with tab2:
        st.header("📄 Upload Legal Document")
        st.markdown("Upload a PDF document for analysis and quick search (temporary - not persisted).")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file:
            processor = PDFProcessor()
            
            with st.spinner("Processing PDF..."):
                # Extract text with page tracking
                pdf_bytes = uploaded_file.read()
                pages_data = processor.extract_text_with_pages(pdf_bytes)
                
                if not pages_data:
                    st.error("Could not extract text from PDF.")
                else:
                    # Get full text for analysis
                    text = ' '.join([p['text'] for p in pages_data])
                    
                    # Analyze document
                    analysis = processor.analyze_document_structure(text)
                    
                    st.success(f"✓ Document processed: {analysis['total_words']} words")
                    
                    # Show analysis
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Words", f"{analysis['total_words']:,}")
                    col2.metric("Pages", len(pages_data))
                    col3.metric("Type", analysis['document_type'])
                    
                    # Extract citations
                    citations = processor.extract_citations_from_pdf(text)
                    if citations:
                        st.markdown(f"**Found {len(citations)} citations in document**")
                        with st.expander("Show citations"):
                            for cite in list(citations)[:10]:
                                st.text(f"• {cite}")
                    
                    # Add to system
                    st.markdown("---")
                    
                    if st.button("💾 Add to Search Index (Session)", type="primary"):
                        with st.spinner("Adding document to search system..."):
                            # Chunk the document with page tracking
                            chunks_data = processor.chunk_pdf_with_pages(pages_data, chunk_size=500, overlap=50)
                            
                            # Add to session state
                            st.session_state.uploaded_docs.append({
                                'filename': uploaded_file.name,
                                'chunks': chunks_data,
                                'num_chunks': len(chunks_data),
                                'total_words': analysis['total_words'],
                                'citations': list(citations) if citations else []
                            })
                            
                            # Rebuild index and update graph
                            rebuild_uploaded_index(system)
                            add_uploaded_docs_to_graph(system)
                            
                            st.success(f"✅ Document added to search system!")
                            st.info(f"📄 {uploaded_file.name}: {len(chunks_data)} chunks created")
                            st.info("🔍 You can now query this document in the Query tab!")
                            st.balloons()
                            
                    st.markdown("---")
                    st.subheader("📄 Document Content Preview")
                    with st.expander("View full text"):
                        st.text(text[:5000] + ("..." if len(text) > 5000 else ""))
    
    with tab3:
        st.header("ℹ️ About KG-CiteRAG")
        
        st.markdown("""
        ### Core Features
        
        1. **Verification Loop**: Every citation is fact-checked against the Knowledge Graph
        2. **Hybrid Retrieval**: Combines semantic search with graph-based relevance
        3. **Dynamic Integration**: Upload new documents and integrate them instantly
        
        ### How It Works
        
        1. **Input**: You ask a question or upload a document
        2. **Hybrid Retrieval**: System searches both by text similarity and citation network
        3. **Generation**: LLM creates an answer with citations
        4. **Verification**: Citations are checked against 75 years of SC precedent
        5. **Output**: Verified answer with validity flags
        
        ---
        
        Built for accurate, verifiable legal research. 🏛️
        """)


if __name__ == "__main__":
    main()
