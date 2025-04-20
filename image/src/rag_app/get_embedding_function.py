# from langchain_aws import BedrockEmbeddings


# def get_embedding_function():
#     embeddings = BedrockEmbeddings()
#     return embeddings


from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small, efficient model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
