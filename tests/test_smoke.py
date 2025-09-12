from app.rag import format_context

def test_format_context_no_crash():
    from langchain.schema import Document
    docs = [Document(page_content="hello world", metadata={"target_name":"misc"})]
    ctx = format_context(docs)
    assert "hello world" in ctx
