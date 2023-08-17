# Multi Document Reader and Chatbot using LangChain and OpenAI

## Summary
Provided here are a few python scripts to help get started with building your own multi document reader and chatbot.
The scripts increase in complexity and features, as follows:

`single-doc.py` Can handle interacting with a single pdf. Sends the entire document content to the LLM prompt.

`single-long-doc.py` Can handle interacting with a long single pdf. Uses embeddings and a vector store to handle
sending only relevant information to the LLM prompts.

`multi-doc-chatbot.py` Can handle interacting with multiple different documents and document types (.pdf, .dox, .txt), 
and remembers the chat history and recent conversations.
It uses embeddings and vector stores to send the relevant information to the LLM prompt. Also provides a chat interface
via the terminal using stdin and stdout. Press `q` to escape the chat window.

[smaameri](https://github.com/smaameri) wrote an article which explores some of the concepts here, as well as walks through building each of the scripts.
[Can read that here](https://medium.com/@ssmaameri/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339)


