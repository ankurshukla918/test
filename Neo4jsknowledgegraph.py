import os
import neo4j
import pandas as pd
from dotenv import load_dotenv
from neo4j_graphrag.llm import AzureOpenAILLM as LLM
from neo4j_graphrag.embeddings.azure_openai import AzureOpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validate configuration
if not all([AZURE_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("❌ One or more environment variables are missing!")

neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI,
                                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize Azure OpenAI LLM for KG Builder
ex_llm = LLM(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version="2024-02-15-preview",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
)

# Initialize Azure OpenAI Embeddings
embedder = AzureOpenAIEmbeddings(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_version="2024-02-15-preview"
)

# ============== DATA LOADING STARTS HERE ==============

# 1. Read CSV file and convert to text
csv_file_path = 'precision-med-for-lupus.csv'
df = pd.read_csv(csv_file_path)

# Convert CSV to text format for KG building
csv_text = df.to_string()

# 1. Build KG and Store in Neo4j Database
kg_builder_csv = SimpleKGPipeline(
    llm=ex_llm,
    driver=neo4j_driver,
    embedder=embedder,
    from_pdf=False
)
await kg_builder_csv.run_async(text=csv_text, document_metadata={"path": csv_file_path})

# ============== DATA LOADING ENDS HERE ==============

# 2. KG Retriever
vector_retriever = VectorRetriever(
    neo4j_driver,
    index_name="text_embeddings",
    embedder=embedder
)

# 3. GraphRAG Class with Azure OpenAI
llm = LLM(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version="2024-02-15-preview"
)
rag = GraphRAG(llm=llm, retriever=vector_retriever)

# 4. Run
response = rag.search("How is precision medicine applied to Lupus?")
print(response.answer)