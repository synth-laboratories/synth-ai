"""
Return the LangfuseGeneration (or, for OpenAI’s drop-in wrapper, the current OTEL span) while you’re still inside it—it already contains input, output, usage, cost, IDs, etc.  ￼

from langfuse import get_client
from langfuse.openai import openai   # drop-in replacement
from opentelemetry import trace

client   = openai.OpenAI()
langfuse = get_client()

with langfuse.start_as_current_generation(name="llm_call",
                                          model="gpt-4o",
                                          input={"messages":[{"role":"user","content":"Hi"}]}) as gen:
    resp = client.chat.completions.create(model="gpt-4o",
                                          messages=[{"role":"user","content":"Hi"}])
    gen.update(output=resp.choices[0].message.content)

    # pull the record out of memory ─ option 1
    llm_call_record = gen                # LangfuseGeneration object
    # or primitives
    llm_call_record_dict = gen.model_dump()

# drop-in wrapper only: inside the same context you can also do
current_span     = trace.get_current_span()      # OTEL span for this generation
llm_attrs        = current_span.attributes       # dict with prompt, completion, tokens, cost, etc.

Yes. During the lifetime of an OpenAI call made through langfuse.openai, that call’s generation span is pushed onto the OTEL context stack and becomes the “current” span, so

from opentelemetry import trace
current = trace.get_current_span()  # == generation span for this LLM call

returns exactly that span while the call (or the surrounding with start_as_current_generation(...)) is still open. When the span ends it’s popped off the stack; afterwards get_current_span() yields its parent (or an InvalidSpan if none), so persist a reference if you need it later.  ￼ ￼
"""