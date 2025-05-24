# src/tools/house_sales_tool.py
from langchain_core.tools import BaseTool
from typing import Type, Optional, Any, List
from pydantic.v1 import BaseModel, Field 

from knowledge_base.vector_store import VectorDB 
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder 
cross_encoder_model_instance = None

class HouseSalesSearchInput(BaseModel):
    query: str = Field(description="The user's question or search query about house sales.")

class HouseSalesInfoTool(BaseTool):
    name: str = "house_sales_information_retriever"
    description: str = (
        "Useful for finding descriptive information about King County house sales listings based on a natural language query. "
        "This includes property details, features, textual descriptions, and locations. "
        "Use this if the user asks for general information, features of a house, or to find houses matching certain criteria. "
        "If the user provides a specific property ID, you can include it in the query to try and find that specific listing's description."
    )
    args_schema: Type[BaseModel] = HouseSalesSearchInput
    
    vector_db: VectorDB 
    llm_for_synthesis: Any 
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
    top_n_retrieval: int = 10
    top_k_rerank: int = 3    

    def _load_cross_encoder(self):
        """Loads the cross-encoder model, caching it globally within this module."""
        global cross_encoder_model_instance
        if cross_encoder_model_instance is None:
            print(f"Loading CrossEncoder model: {self.cross_encoder_model_name}...")
            try:
                cross_encoder_model_instance = CrossEncoder(self.cross_encoder_model_name)
                print("CrossEncoder model loaded successfully.")
            except Exception as e:
                print(f"Error loading CrossEncoder model: {e}")
        return cross_encoder_model_instance

    def _run(self, query: str) -> str:
        if not self.vector_db.is_ready(): 
            return "Error: The house sales database (VectorDB) is not ready or embedding model is not loaded."
        if not self.llm_for_synthesis: 
            return "Error: LLM for synthesis is not available in the HouseSalesInfoTool."

        print(f"\n<<< HouseSalesInfoTool activated with query: {query} >>>")
        try:
            print(f"Initial retrieval: Fetching top {self.top_n_retrieval} documents for query: '{query}'")
            initial_results = self.vector_db.search_points(
                query_text=query, 
                top_k=self.top_n_retrieval 
            )
        except Exception as e:
            return f"Error during initial database search in HouseSalesInfoTool: {e}"

        if not initial_results:
            return "No initial information found in the house sales vector database for that query."
        
        print(f"Initial retrieval found {len(initial_results)} documents.")
        cross_encoder = self._load_cross_encoder()
        if cross_encoder:
            print("Re-ranking retrieved documents with CrossEncoder...")
            sentence_pairs = []
            for hit in initial_results:
                doc_text = hit.payload.get("description", "") 
                if doc_text:
                    sentence_pairs.append([query, doc_text])
            
            if sentence_pairs:
                try:
                    scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False)
                    reranked_results_with_scores = []
                    for i in range(len(initial_results)):
                        if i < len(scores):
                             reranked_results_with_scores.append({
                                "hit": initial_results[i], 
                                "cross_encoder_score": scores[i]
                            })
                    reranked_results_with_scores.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
                    final_hits = [item["hit"] for item in reranked_results_with_scores[:self.top_k_rerank]]
                    print(f"Re-ranking complete. Selected top {len(final_hits)} documents.")
                except Exception as e:
                    print(f"Error during cross-encoder prediction or sorting: {e}. Falling back to initial results.")
                    final_hits = initial_results[:self.top_k_rerank] 
            else:
                print("No valid sentence pairs for re-ranking. Using initial results.")
                final_hits = initial_results[:self.top_k_rerank] 
        else:
            print("CrossEncoder model not available. Skipping re-ranking. Using initial results.")
            final_hits = initial_results[:self.top_k_rerank] 

        if not final_hits:
             return "No information found after attempting to refine search results."
        context_parts = ["Relevant information found in house sales listings (refined search):\n"]
        for i, hit in enumerate(final_hits):
            payload = hit.payload
            desc = payload.get("description", f"House ID {hit.id} with limited descriptive data.")
            ce_score_info = ""
            if cross_encoder: 
                original_item = next((item for item in reranked_results_with_scores if item["hit"].id == hit.id), None)
                if original_item:
                    ce_score_info = f"(Relevance Score: {original_item['cross_encoder_score']:.2f})"

            context_parts.append(f"Listing {i+1} (ID: {hit.id}, Qdrant Score: {hit.score:.2f}) {ce_score_info}:\n{desc}\n")
        context_str = "\n".join(context_parts)

        synthesis_prompt = f"""Based ONLY on the following context about house sales listings, answer the user's original question.
If the context doesn't provide an answer, state that the specific details were not found in the descriptions. Do not make up information.

Context:
{context_str}

User's Original Question: "{query}"

Answer:"""
        
        try:
            response = self.llm_for_synthesis.invoke([HumanMessage(content=synthesis_prompt)])
            answer = response.content.strip()
            print(f"HouseSalesInfoTool (with re-ranking) synthesized answer: {answer[:200]}...")
            return answer
        except Exception as e:
            print(f"HouseSalesInfoTool LLM synthesis error (post re-ranking): {e}")
            return f"Error synthesizing answer from re-ranked house sales search results: {e}"

    async def _arun(self, query: str) -> str:
        print(f"\n<<< HouseSalesInfoTool ASYNC (with re-ranking) activated with query: {query} >>>")
        return self._run(query) 