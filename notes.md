# RAG

## Main Components
### Indexing
[Document Loading:]

   * Text Representation:
       * Retrieve documents that relates 
         to the input question
 
>        questioen -> Retriever -> [Document]
>                       |
>                 Load Documents
>                       |
>                   Documents

   * Numerical Representation:
       * Numerical Representation of Document
         because it's easier to compare vectors
         with numbers then free form text

>   (x,y,z..) -> Cosine similarity, etc -> [(x,y,z..)]
>                       |
>                 Load Documents
>                       |
>                   (x,y,z..)
>                   [(x,y,z..)]

[Statical and Machine Learned Representations:]

   * Statical:
       * Bag of Words Representation Search:

   * Machine Learned: 
       * Embedding: 

### Retrieval

### Generation
