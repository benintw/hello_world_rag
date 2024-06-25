# hello_world_rag




# RAG Implementation with Ollama and LangChain
 
This repository contains an implementation of the Retrieval-Augmented Generation (RAG) model using Open AI's LLaMA large language model and the LangChain library.
 
## Overview
 
The code instantiates an OLLAMA model, loads Wikipedia content, splits the text into manageable chunks, creates sentence embeddings with SentenceTransformers, and builds a vector store using Chroma. Finally, it creates a QA chain using the OLLAMA model and the vector store retriever.
 
## Usage
 
To use this code, simply run the Python script. The output will be the generated response to a given question.
 
## Example Question
 
The example question used in this implementation is: "What is Tsai's energy policy?"
 
## Dependencies
 
 
LangChain
 
OLLAMA
 
SentenceTransformers
 
Chroma
 
 
## Notes
 
This is my first attempt at implementing RAG using OLLAMA and LangChain. While the code is functional, it may not be optimized for performance or scalability. Further improvements and testing are needed to ensure the model's reliability.
 
I hope this helps! Let me know if you have any questions or need further assistance.
