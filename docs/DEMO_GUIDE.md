# Final live check and short demonstration

Run `python scripts/check_project.py` first. Keep the API key off-screen. Open the app with `python -m streamlit run app.py`.

Use real responses produced during the demonstration. Their content, dates and tool values can differ from the historical examples. Avoid repeated connection checks: the following conversation demonstrates the required workflow directly.

## 1. RAG destination answer

Ask: **Which local dishes should I try at Singapore hawker centres?**

Show the chat answer, source links and evidence details. Explain: “This answer searches the saved Singapore travel documents. The citations show where its information came from. No weather or currency call is needed.”

## 2. Currency through the custom MCP server

Ask: **Convert 10000 INR to SGD using the latest available reference rate.**

Show `convert_currency`, its actual arguments, returned amount, reference-rate date and provider. Explain: “The model chose our currency tool. Our MCP client calls the server we wrote, and that server gets the reference rate. It is an estimate before bank fees.”

## 3. Required combined scenario

Start a new chat. Ask:

**Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast. We are two adults travelling without children. I am vegetarian and prefer public transport.**

Show all three dated days, the weather table, sheltered main plans when indicated, transport/food guidance and citations. Explain: “The destination ideas come from our documents. The current forecast comes through our weather MCP tool. The assistant combines them into a daily plan and keeps our preferences.”

Download this Markdown and JSON. Check that every day has a supported main activity and that the weather covers the same dates. A `partial` answer with an unresolved main day should be reviewed before using it as the final successful example.

## 4. Conversational context

In the same chat ask: **Make it four days instead. Keep my food and transport preferences, and keep the second day indoors.**

Show four dates, retained vegetarian/public-transport preferences, Day 2, and the updated weather tool input. Explain: “I did not repeat my food and transport preferences. They remain in the conversation, while the new four-day request replaces the earlier length.”

Download the answer and JSON. Review the full result before recording it as a passed context example.

## 5. Missing information, if needed for the video

Ask an out-of-scope question such as **Which ski resorts should I visit in Switzerland?** The assistant should state that its knowledge base covers Singapore. This is one additional model request; the saved historical example can be shown instead if clearly labelled recorded.

## 6. Brief code walkthrough

Show these files:

- `data/sources.json`: four public resources and source metadata.
- `src/chunk_documents.py`, `src/embeddings.py`, `src/retriever.py`: chunking, the same embedding model for documents/queries, semantic retrieval and reranking.
- `mcp_server/server.py` and `src/mcp_client.py`: the candidate's server, two tools, stdio connection and discovery.
- `src/assistant.py`: LangChain prompt, model-selected tools and bounded correction.
- `src/conversation_memory.py`: ordered preferences and recent exchanges.
- `README.md`: setup, architecture and limitations.

## Recording and submission

Record the working interface plus the short explanation above. Save the video and selected real answer downloads. Commit the final code to your Git repository and use the submission format requested by the evaluator. The repaired working ZIP includes historical diagnostics for review; select the clean examples for the final submission. Follow the source acquisition/redistribution notes.

A typical sequence is one RAG model call, two currency calls, two combined calls and two follow-up calls, with at most one extra correction per message. Actual usage depends on the returned responses. Provider quota/capacity errors cannot be eliminated by local code.
