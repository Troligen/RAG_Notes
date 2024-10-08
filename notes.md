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

statical Example: 

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

This is interesting because LLMs in general including
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

Interesting note here is that the length
of the vector is 1536 which is a static length
so both the document and question
are both computed to a 1536 dimensional vector.
So the 1536 vector encodes the symantics 
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
> the document easy to retrieve

[How to think about it]
> Imagein a 3d space -  we embedd the doccument somwere in 3d space.
> We then make a search query and embedd the search query in 3d space.
> Afterwards we make a similarity surch with the embedded query
> And this allowes os to retriewe the closest 1 - 1000000+ documents
> that matches the search query. 
> A point in space is also an representation of the semantic meaning
> This means that points in the same space that are close to eachother 
> naturally have a similar semantic meaning which means it likely uf not
> garanteed to be related to each other in some way
> This Idea and thechnolegy is the cornerstone for a lot of search and retrieval
> methods u see using modern vector stores.

[Why does this work]
> By embedding a chunk of a file we represent the symantic meanig of
> that chunk by giving it a position in an x dimensional space by which is the vector.
> When we then embedd the query the query in questions also get's it symantic meanig
> reprecented as a postion in the same x dimensional space. This gives us the
> possibility to compare the simelarity in the querys position in space
> with all the chunkes we've embedded position in space and then retrieve the closest 
> x neibours of the querys position in space. 

### Generation
[Basic workflow]
First, this introduces the notion of a prompt with placeholder keys where we later can
inject information like context, documents or user input. 
A prompt can also be used to alter the the query that's used to retrieve the dcuments
that's stored in our vectorstore. 

[example prompt]
```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
```
> A sysmtem prompt which injects the context of the retrieved documents. 


```python
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
```
> A system prompt designed to alter the query used to fetch from the vectorstore
> Worth noting: If it's the first message the prompt is designed to do nothing -
> this prompt is also only meant as a middle man, it will only alter the query and not
> the actual message that will be sent to the llm
> Read more about this under [Query-Translation]

## Query Translation

[What is Query Translations?]
Query Translations is the first stage of an advanced RAG-pipeline
and the goal of query translation is to take an input user question and
then translate it in some way in order to improve retrieval

[Why Query Translation?]
A users query can be ambiguous and poorly written. This cause problems
since the goal is to do some kind of similarity search between the query
and the vectorstore and if the query is badly written the similarity search
might give the wrong documents. 

[Aproches to solve this]
* High abstraction
    * Step-back question
        * Step-back promoting

* Mid abstraction
    * Query rewriting
        * Multi-query
        * RAG-Fusion

* Low abstraction
    * Sub-Question
        * Least-to-most
        * IR-CoT

### Multi-Query
[Main Idea]
The main idea is that we take the user question and break it down to
diferently worded questions from different perspektives. The intuition here
is that it is possible that the way that a question is initially worded once 
embedded is not well aligned or in close proximity in this High dimensional embedding
space to a document we want to retrieve. So by rewriting the question in different
ways can we then combine it with retrieval by comparing the retrieved documents
or combine them in some way that fits the question and then use generation to perform RAG. 

[Example Prompt for Multi-Query]
```python
template = """
You are an AI language model assistan. Your task is to generate five
diferent versions of the given user question to retrieve relevant documents from a
vector databse. By generating multiple perspectives on the user question, your goal 
is to help the user overcome some of the limitatons of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}
"""
```

[Example: How it's done]
> Ask question -> Send it to an llm -> llm generates n different version of question
> -> We do a similarity search on all n versions of questions -> We the smuch the list together -
> and remove duplicates of retrieved documents -> We then send the original question with the -
> retrieved documents to the main llm. 

[Example Code Using LangChain]
```python
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattend_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattend_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "what is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retrieve.map() | get_unique_union

# RAG 
template = """
Answer the following question based onthis context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperatur=0)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"quesion":question})
```

### RAG-fusion
[Main Idea]
It follows the sama main idea as Multy RAG where we take the
user query and make a model reformulate it into different versions
but we then apply a ranking step over our documents. 


[RAG Fusion Prompt Example]
```python
from langchain.prompt import ChatPromptTemplate

# RAG-Fusion: Relate
template = """
You are a hekpful assistant that generates multiple search queries based of a single input query. \n
Generate multiple search queries related to: {question} \n
output: {4 queries}:
"""
prompt_rag_fusion = ChatPomptTemplate.from_tamplate(tamplate)
```

[Example Code using LangChain]
```python
from langchain_core.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langhcain.load import dumps, loads

generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not in the fused_scores dictionary, add it with an inital score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any 
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based of their ranked score in decending order to get the final reranked result
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rankfusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
```

[Final RAG chain]
```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt =ChatPromptTemplate.from_template(template)

finaö_rag_chain = (
    {"context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question": question})
```
### Least-to-most
The object is to first take a question and then decompose it to 
a set of subproblems

[Example]
Problem to solve: 
Last letter concatenation - The goal is to take a question
"think, machine, learning" and then get to the answer "keg"
using an least-to-most aproch

Decompose the problem:
Q: "think, machine, learning"
A: "think" "think, machine" "think, machine, learning"

This is a least-to-most prompt context (decomposition) for the
last-leter-concatenation task. What it does is that an llm takes the question
which is "think, machine, learning" and decompose it to 3 different 
sub problems that it will use to solve the main problem

Figuring out the answer:
The LLM will now take the different subproblems and solv them indevidually 
and use the answer of the previous problem to easier solve the problem after.
(ignoring the first one since it's only 1 word)

Q1: "think, machine"
A1: The last letter of "think" is "k", the last letter of "machine" is "e", 
    this gives us ke. So the output is "ke"

Q2: "think, machine, learning"
A2: "think, machine" outputs "ke", The last letter of "learing" is "g",
    concatenating "ke" and "g" leads to "keg". So the output is "keg"

### IR-CoT



