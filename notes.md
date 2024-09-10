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

### Retrieval

### Generation
