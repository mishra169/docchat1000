Project: A chatbot based on LLM and Knowledge Graphs capable of reading over 1000 documents.

Knowledge Graphs:Knowledge graphs are powerful frameworks for organizing, linking, and sharing data with universal meaning.
In Resource Description Framework, data is represented as triples that consist of subject, predicate, and object. These are known RDF statements, i.e. “Alice is a Friend of Jack.” These triples form a directed graph, with the subject and object as nodes and the predicate as labeled edge. LPG uses nodes to represent data elements and labeled edges to connect nodes. Each edge represents a relationship between nodes, and nodes can have properties associated with them.
![image](https://github.com/mishra169/docchat1000/assets/104723673/fcd5cdbb-936b-4650-a1f4-51ea43ba6973)
Knowledge graphs can become very large, with millions or billions of interconnected elements. RDF databases are designed to handle this level of scale, and some of them can distribute the work across multiple servers to make it faster. This scalability is crucial for organizations dealing with large volumes of distributed data. On the other hand, labeled property graphs might face challenges in managing and analyzing massive databases.
RDF aligns with the ideas and protocols behind Linked Data. Specifically, RDF can enable the connection of data from different sources using unique identifiers without massive integration overhead . This interconnectedness allows for a vast network of knowledge that spans across various domains – a key requirement for knowledge graphs. While labeled property graphs can also link data, RDF’s standardized approach provides more opportunities for cross-domain interoperability and integration.  

For implementing the Knowledge graph in this project we have used RAPTOR-RECURSIVE ABSTRACTIVE PROCESSING
FOR TREE-ORGANIZED RETRIEVAL. To know more about the RAPTOR and graphs tree refer to https://arxiv.org/pdf/2401.18059

Other used tools:
Llama GPT: LlamaGPT. A self-hosted, offline, ChatGPT-like chatbot, powered by Llama 2. 100% private, with no data leaving your device.
