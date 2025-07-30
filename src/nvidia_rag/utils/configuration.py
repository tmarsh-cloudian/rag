# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The definition of the application configuration."""

import os
from .configuration_wizard import ConfigWizard
from .configuration_wizard import configclass
from .configuration_wizard import configfield


@configclass
class VectorStoreConfig(ConfigWizard):
    """Configuration class for the Vector Store connection.

    :cvar name: Name of vector store
    :cvar url: URL of Vector Store
    """

    name: str = configfield(
        "name",
        default="milvus",
        help_txt="The name of vector store",  # supports milvus
    )
    url: str = configfield(
        "url",
        default="http://localhost:19530",
        help_txt="The host of the machine running Vector Store DB",
    )
    nlist: int = configfield(
        "nlist",
        default=64,
        help_txt="Number of cluster units",  # IVF Flat milvus
    )
    nprobe: int = configfield(
        "nprobe",
        default=16,
        help_txt="Number of units to query",  # IVF Flat milvus
    )
    index_type: str = configfield(
        "index_type",
        default="GPU_CAGRA",
        help_txt="Index of the vector db",  # IVF Flat for milvus
    )

    enable_gpu_index: bool = configfield(
        "enable_gpu_index",
        default=True,
        help_txt="Flag to control GPU indexing",
    )

    enable_gpu_search: bool = configfield(
        "enable_gpu_search",
        default=True,
        help_txt="Flag to control GPU search",
    )

    search_type: str = configfield(
        "search_type",
        default="dense", # dense or hybrid
        help_txt="Flag to control search type - 'dense' retrieval or 'hybrid' retrieval",
    )

    default_collection_name: str = configfield(
        "default_collection_name",
        default="multimodal_data",
        env_name="COLLECTION_NAME",
        help_txt="Default collection name for vector store",
    )

    consistency_level: str = configfield(
        "consistency_level",
        default="Strong", # "Bounded", "Strong", "Session"
        env_name="APP_VECTORSTORE_CONSISTENCYLEVEL",
        help_txt="Consistency level for vector store",
    )


@configclass
class NvIngestConfig(ConfigWizard):
    """
    Configuration for NV-Ingest.
    """
    # NV-Ingest Runtime Connectivity Configuration parameters
    message_client_hostname: str = configfield(
        "message_client_hostname",
        default="localhost", # TODO
        help_txt="NV Ingest Message Client Host Name",
    )

    message_client_port: int = configfield(
        "message_client_port",
        default=7670,
        help_txt="NV Ingest Message Client Port",
    )

    # Extraction Configuration Parameters (Add additional parameters here)
    extract_text: bool = configfield(
        "extract_text",
        default=True,
        help_txt="Enable extract text for nv-ingest extraction",
    )

    extract_infographics: bool = configfield(
        "extract_infographics",
        default=False,
        help_txt="Enable extract infographics for nv-ingest extraction",
    )

    extract_tables: bool = configfield(
        "extract_tables",
        default=True,
        help_txt="Enable extract tables for nv-ingest extraction",
    )

    extract_charts: bool = configfield(
        "extract_charts",
        default=True,
        help_txt="Enable extract charts for nv-ingest extraction",
    )

    extract_images: bool = configfield(
        "extract_images",
        default=False,
        help_txt="Enable extract images for nv-ingest extraction",
    )

    pdf_extract_method: str = configfield(
        "pdf_extract_method",
        default="None", # Literal['pdfium','nemoretriever_parse','None']
        help_txt="Extract method 'pdfium', 'nemoretriever_parse', 'None'",
    )

    text_depth: str = configfield(
        "text_depth",
        default="page", # Literal['page', 'document']
        help_txt="Extract text by 'page' or 'document'",
    )

    # Splitting Configuration Parameters (Add additional parameters here)
    tokenizer: str = configfield(
        "tokenizer",
        default="intfloat/e5-large-unsupervised", # Literal["intfloat/e5-large-unsupervised" , "meta-llama/Llama-3.2-1B"]
        help_txt="Tokenizer for text splitting.",
    )

    chunk_size: int = configfield(
        "chunk_size",
        default=1024,
        help_txt="Chunk size for text splitting.",
    )

    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=150,
        help_txt="Chunk overlap for text splitting.",
    )

    # Captioning Configuration Parameters
    caption_model_name: str = configfield(
        "caption_model_name",
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        help_txt="NV Ingest Captioning model name",
    )

    caption_endpoint_url: str = configfield(
        "caption_endpoint_url",
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        help_txt="NV Ingest Captioning model Endpoint URL",
    )

    enable_pdf_splitter: bool = configfield(
        "enable_pdf_splitter",
        default=True,
        help_txt="Enable post chunk split for NV Ingest",
    )


@configclass
class ModelParametersConfig(ConfigWizard):
    """Configuration class for model parameters based on model name.

    This defines default parameters for different LLM models.
    """

    max_tokens: int = configfield(
        "max_tokens",
        default=1024,
        help_txt="The maximum number of tokens to generate in any given call."
    )

    temperature: float = configfield(
        "temperature",
        default=0.2,
        help_txt="The sampling temperature to use for text generation."
    )

    top_p: float = configfield(
        "top_p",
        default=0.7,
        help_txt="The top-p sampling mass used for text generation."
    )


@configclass
class LLMConfig(ConfigWizard):
    """Configuration class for the llm connection.

    :cvar server_url: The location of the llm server hosting the model.
    :cvar model_name: The name of the hosted model.
    """

    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The location of the Triton server hosting the llm model.",
    )
    model_name: str = configfield(
        "model_name",
        default="nvidia/llama3.1-nemotron-nano-4b-v1.1",
        help_txt="The name of the hosted model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are nvidia-ai-endpoints",
    )
    model_name_pandas_ai: str = configfield(
        "model_name_pandas_ai",
        default="ai-mixtral-8x7b-instruct",
        help_txt="The name of the ai catalog model to be used with PandasAI agent",
    )
    # Add model parameters configuration
    parameters: ModelParametersConfig = configfield(
        "parameters",
        env=False,
        help_txt="Model-specific parameters for generation.",
        default=ModelParametersConfig(),
    )

    def get_model_parameters(self) -> dict:
        """Return appropriate parameters based on the model name.

        Returns a dictionary with max_tokens, temperature, and top_p
        adjusted according to the model name.
        """
        params = {
            "max_tokens": self.parameters.max_tokens,
            "temperature": self.parameters.temperature,
            "top_p": self.parameters.top_p
        }

        # Check for deepseek model
        if "deepseek-r1" in str(self.model_name):
            params["max_tokens"] = 128000
            params["temperature"] = 0.6
            params["top_p"] = 0.95

        # Check for llama model
        if "llama-3.3-nemotron-super-49b" in str(self.model_name):
            if os.getenv("ENABLE_NEMOTRON_THINKING", "false").lower() == "true":
                params["max_tokens"] = 32768
                params["temperature"] = 0.6
                params["top_p"] = 0.95
            else:
                params["max_tokens"] = 32768
                params["temperature"] = 0
                # TODO: Add support to pass None as top_p
                params["top_p"] = 0.1

        return params


@configclass
class QueryRewriterConfig(ConfigWizard):
    """Configuration class for the Query Rewriter.
    """
    model_name: str = configfield(
        "model_name",
        default="meta/llama-3.1-8b-instruct",
        help_txt="The llm name of the query rewriter model",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The location of the query rewriter model.",
    )
    enable_query_rewriter: bool = configfield(
        "enable_query_rewriter",
        env_name="ENABLE_QUERYREWRITER",
        default=False,
        help_txt="Enable query rewriter",
    )
    # TODO: Add temperature, top_p, max_tokens


@configclass
class TextSplitterConfig(ConfigWizard):
    """Configuration class for the Text Splitter.

    :cvar chunk_size: Chunk size for text splitter. Tokens per chunk in token-based splitters.
    :cvar chunk_overlap: Text overlap in text splitter.
    """

    model_name: str = configfield(
        "model_name",
        default="Snowflake/snowflake-arctic-embed-l",
        help_txt="The name of Sentence Transformer model used for SentenceTransformer TextSplitter.",
    )
    chunk_size: int = configfield(
        "chunk_size",
        default=510,
        help_txt="Chunk size for text splitting.",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        default=200,
        help_txt="Overlapping text length for splitting.",
    )


@configclass
class EmbeddingConfig(ConfigWizard):
    """Configuration class for the Embeddings.

    :cvar model_name: The name of the huggingface embedding model.
    """

    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.2-nv-embedqa-1b-v2",
        help_txt="The name of huggingface embedding model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are hugginface",
    )
    dimensions: int = configfield(
        "dimensions",
        default=2048,
        help_txt="The required dimensions of the embedding model. Currently utilized for vector DB indexing.",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The url of the server hosting nemo embedding model",
    )


@configclass
class RankingConfig(ConfigWizard):
    """Configuration class for the Re-ranking.

    :cvar model_name: The name of the Ranking model.
    """

    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        help_txt="The name of Ranking model.",
    )
    model_engine: str = configfield(
        "model_engine",
        default="nvidia-ai-endpoints",
        help_txt="The server type of the hosted model. Allowed values are nvidia-ai-endpoints",
    )
    server_url: str = configfield(
        "server_url",
        default="",
        help_txt="The url of the server hosting nemo Ranking model",
    )
    enable_reranker: bool = configfield(
        "enable_reranker",
        env_name="ENABLE_RERANKER",
        default=True,
        help_txt="Enable reranking",
    )


@configclass
class RetrieverConfig(ConfigWizard):
    """Configuration class for the Retrieval pipeline.

    :cvar top_k: Number of relevant results to retrieve.
    :cvar score_threshold: The minimum confidence score for the retrieved values to be considered.
    """

    top_k: int = configfield(
        "top_k",
        default=10,
        help_txt="Number of relevant results to retrieve",
    )
    vdb_top_k: int = configfield(
        "vdb_top_k",
        env_name="VECTOR_DB_TOPK",
        default=100,
        help_txt="Number of relevant results to retrieve from vector db",
    )
    score_threshold: float = configfield(
        "score_threshold",
        default=0.25,
        help_txt="The minimum confidence score for the retrieved values to be considered",
    )
    nr_url: str = configfield(
        "nr_url",
        default='http://retrieval-ms:8000',
        help_txt="The nemo retriever microservice url",
    )
    nr_pipeline: str = configfield(
        "nr_pipeline",
        default='ranked_hybrid',
        help_txt="The name of the nemo retriever pipeline one of ranked_hybrid or hybrid",
    )

@configclass
class TracingConfig(ConfigWizard):
    """Configuration class for Open Telemetry Tracing.
    """
    enabled: bool = configfield(
        "enabled",
        default=False,
        help_txt="Enable Open Telemetry Tracing",
    )
    otlp_http_endpoint: str = configfield(
        "otlp_http_endpoint",
        default="",
        help_txt=""
    )
    otlp_grpc_endpoint: str = configfield(
        "otlp_grpc_endpoint",
        default="",
        help_txt=""
    )

@configclass
class VLMConfig(ConfigWizard):
    """Configuration class for the VLM.
    """
    server_url: str = configfield(
        "server_url",
        default="http://localhost:8000/v1",
        help_txt="The url of the server hosting the VLM model",
    )
    model_name: str = configfield(
        "model_name",
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        help_txt="The name of the VLM model",
    )

@configclass
class MinioConfig(ConfigWizard):
    """Configuration class for the Minio.
    """
    endpoint: str = configfield(
        "endpoint",
        env_name="MINIO_ENDPOINT",
        default="localhost:9010",
        help_txt="The endpoint of the minio server",
    )
    # TODO: Hide secret keys so it's not visible when showing config
    access_key: str = configfield(
        "access_key",
        env_name="MINIO_ACCESSKEY",
        default="minioadmin",
        help_txt="The access key of the minio server",
    )
    secret_key: str = configfield(
        "secret_key",
        env_name="MINIO_SECRETKEY",
        default="minioadmin",
        help_txt="The secret key of the minio server",
    )

@configclass
class SummarizerConfig(ConfigWizard):
    """Configuration class for the Summarizer.
    """
    model_name: str = configfield(
        "model_name",
        env_name="SUMMARY_LLM",
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        help_txt="The name of the summarizer model",
    )
    server_url: str = configfield(
        "server_url",
        env_name="SUMMARY_LLM_SERVERURL",
        default="",
        help_txt="The url of the server hosting the summarizer model",
    )
    max_chunk_length: int = configfield(
        "max_chunk_length",
        env_name="SUMMARY_LLM_MAX_CHUNK_LENGTH",
        default=50000,
        help_txt="Maximum chunk size in characters for the summarizer model",
    )
    chunk_overlap: int = configfield(
        "chunk_overlap",
        env_name="SUMMARY_CHUNK_OVERLAP",
        default=200,
        help_txt="Overlap between chunks for iterative summarization (in characters)",
    )

@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar vector_store: The configuration of the vector db connection.
    :type vector_store: VectorStoreConfig
    :cvar llm: The configuration of the backend llm server.
    :type llm: LLMConfig
    :cvar text_splitter: The configuration for text splitter
    :type text_splitter: TextSplitterConfig
    :cvar embeddings: The configuration for huggingface embeddings
    :type embeddings: EmbeddingConfig
    :cvar prompts: The Prompts template for RAG and Chat
    :type prompts: PromptsConfig
    """

    vector_store: VectorStoreConfig = configfield(
        "vector_store",
        env=False,
        help_txt="The configuration of the vector db connection.",
        default=VectorStoreConfig(),
    )
    llm: LLMConfig = configfield(
        "llm",
        env=False,
        help_txt="The configuration for the server hosting the Large Language Models.",
        default=LLMConfig(),
    )
    query_rewriter: QueryRewriterConfig = configfield(
        "query_rewriter",
        env=False,
        help_txt="The configuration for the query rewriter.",
        default=QueryRewriterConfig(),
    )
    text_splitter: TextSplitterConfig = configfield(
        "text_splitter",
        env=False,
        help_txt="The configuration for text splitter.",
        default=TextSplitterConfig(),
    )
    embeddings: EmbeddingConfig = configfield(
        "embeddings",
        env=False,
        help_txt="The configuration of embedding model.",
        default=EmbeddingConfig(),
    )
    ranking: RankingConfig = configfield(
        "ranking",
        env=False,
        help_txt="The configuration of ranking model.",
        default=RankingConfig(),
    )
    retriever: RetrieverConfig = configfield(
        "retriever",
        env=False,
        help_txt="The configuration of the retriever pipeline.",
        default=RetrieverConfig(),
    )
    nv_ingest: NvIngestConfig = configfield(
        "nv_ingest",
        env=False,
        help_txt="The configuration for nv-ingest.",
        default=NvIngestConfig(),
    )
    tracing: TracingConfig = configfield(
        "tracing",
        env=False,
        help_txt="",
        default=TracingConfig()
    )
    enable_guardrails: bool = configfield(
        "enable_guardrails",
        env_name="ENABLE_GUARDRAILS",
        default=False,
        help_txt="Enable guardrails",
    )
    enable_citations: bool = configfield(
        "enable_citations",
        env_name="ENABLE_CITATIONS",
        default=True,
        help_txt="Enable citations",
    )
    enable_vlm_inference: bool = configfield(
        "enable_vlm_inference",
        env_name="ENABLE_VLM_INFERENCE",
        default=False,
        help_txt="Enable VLM inference",
    )
    vlm: VLMConfig = configfield(
        "vlm",
        env=False,
        help_txt="The configuration for the VLM.",
        default=VLMConfig(),
    )
    minio: MinioConfig = configfield(
        "minio",
        env=False,
        help_txt="The configuration of the minio server.",
        default=MinioConfig(),
    )
    temp_dir: str = configfield(
        "temp_dir",
        env_name="TEMP_DIR",
        default="./tmp-data",
        help_txt="The temporary directory for the application.",
    )
    summarizer: SummarizerConfig = configfield(
        "summarizer",
        env=False,
        help_txt="The configuration for the summarizer.",
        default=SummarizerConfig(),
    )
