# RAG

## Main Components
### Indexing
[Document Loading:]

   * Text Representation:
       * Retrieve documents that relates 
         to the input question
       
 
>       questioen -> Retriever -> [Document]
>                       |
>                 Load Documents
>                       |
>                   Documents

   * Numerical Representation:
       * Numerical Representation of Document
         because it's easier to compare vectors
         with numbers then free form text
       * Text documents being compressed to
         sequens of numbers for easier searches.

>   (x,y,z..) -> Cosine similarity, etc -> [(x,y,z..)]
>                       |
>                 Load Documents
>                       |
>                   (x,y,z..)
>                   [(x,y,z..)]



[Statical and Machine Learned Representations:]

statistical Example: 

    Spars vectors = Looking at the frequency of words
    and create a Spars Vector in such the vecotor location
    are a large vocabulay of possible words where each value
    represent the number of the occurences of that 
    particualar word and it's spars because
    there's many 0s

>    There's very good search methods over these types
>    of numerical representations. 

Machine Learned Example: 

    Embedding methods = You take and build a compressed fixed
    length reprecentation of a document. Have been developed
    with correspondingly very strong search methods over 
    embeddings. 

* Loading Splitting, and Embedding

    * Splitting:

        * We take a document and split it (chunking)
          because of the limited context window of
          an embedding model 
          ([mayne between 512-8000 tokens])

    * Embedding:

        * Each document is compressed into a vector,
          said vector then captures a symantic meaning 
          of the document and the vector get's indexed.
          Questions can be embdedded in the exact same way
          which then allows for a numerical comparison 
          in some form using a veriaty of different methods
          to fish out the relevent documents relevent to said questions

[Code Example:]
```python
"""
Example: Returning numbers of a tokens in a string

This is interresting because LLMs in general including
embedding operates on tokens so it's nice to understand
how large the documents are that u try to feed in.
Mainly because the embedding has a limited context window.
"""
import tiktoken

question = "What kind of pets do I like?"
documents = "My favorite pet is a cat"

def num_tokens_from_string(string: str, encoding_name: srt) -> inte:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")  
#output: 8
```

```python
"""
Example: embed a question and
a document to a vector embedding 
using LangChain and openAI

Interresting note here is that the length
of the vector is 1536 which is a static length
so both the document and question
are both computed to a 1536 dimensional vector.
So the 1536 vectory encodes the symantics 
of the text that u pass.
"""

from langchain_openai import OpenAIEmbeddings

embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)
#output: 1536
```

```python
"""
Example: cosin simelarity to campare 2 vectors
"""
import numpy as np

def cosine_similarity(vec1: list[int], vec2: list[int]) -> int:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = no.linalf.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print(f"Cosine Similarity: {similarity}")
#Output: Cosine similarity: 0.8812349768269672
```

```python
#### INDEXING ####
"""
Example: Loading documents,
         Splitting document,
         embedd document

What is happening:
Takeing each split, we're embedding it
using openAI embeddings into a vector representation,
And then store it with a link to the raw doccument
itself in our vector store
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.cio/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()


#### SPLIT ####
from langchain.text_spitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chink_size=300,
    chunk_overlap=50
)

#### Make Splits ####
splits = text_splitter.split_documents(blog_docs)

#### Index ####
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vectorestore = Chroma.from_documents(documents=splits,
                                     embedding=OpenAIEmbeddings()

retriever = vectorstore.as_retriever()
```

### Retrieval
[note] 
> The act of indexing also makes 
> the document easy to retriewe

[How to think about it]
> Imagein a 3d space -  we embedd the doccument somwere in 3d space.
> When then make a search query and embedd the search query in 3d space.
> Afterwards we make a similarity surch with the embedded query
> And this allowes os to retriewe the closest 1 - 1000000+ documents
> that matches the search query.

[Why does this work]
> By embedding a chunk of a file we represent the symantic meanig of
> that chunk by giving it a position in an x dimensional space by which is the vector.
> When we then embedd the query the query in questions also get's it symantic meanig
> reprecented as a postion in the same x dimensional space. This gives us the
> possibility to compare the simelarity in the querys position in space
> with all the chunkes we've embedded position in space and then retrieve the closest 
> x neibours of the querys position in space. 

### Generation
