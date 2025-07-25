#!/usr/bin/env python3
"""
Test OTEL-based tracing for Gemini (Google AI).
Demonstrates how to capture traces for providers not directly supported by Langfuse.
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from synth_ai.tracing_v2.session_tracer import SessionTracer, extract_model_info_from_attrs

# Load environment variables
load_dotenv()


def setup_otel_tracing():
    """Set up OpenTelemetry tracing."""
    # Create tracer provider
    provider = TracerProvider()
    
    # Add console exporter for debugging
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    provider.add_span_processor(span_processor)
    
    # Set the tracer provider
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer(__name__)


def test_gemini_with_otel():
    """Test Gemini with manual OTEL tracing."""
    print("=== Testing Gemini with OTEL Tracing ===\n")
    
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found.")
        return None
    
    genai.configure(api_key=api_key)
    
    # Get OTEL tracer
    tracer = setup_otel_tracing()
    
    # Initialize session tracer
    session_tracer = SessionTracer(traces_dir="otel_traces")
    session = session_tracer.start_session("gemini-otel-test")
    timestep = session_tracer.start_timestep("gemini_call")
    
    # Create OTEL span
    with tracer.start_as_current_span("gemini.chat.completions") as span:
        try:
            # Set span attributes
            model_name = "gemini-1.5-flash"
            prompt = "Say hello and count to 3."
            
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.request.provider", "google")
            span.set_attribute("gen_ai.prompt", prompt)
            span.set_attribute("llm.model_name", model_name)
            
            # Track timing
            start_time = time.time()
            
            # Initialize Gemini model
            model = genai.GenerativeModel(model_name)
            
            # Make API call
            response = model.generate_content(prompt)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts (if available)
            prompt_tokens = None
            completion_tokens = None
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
            
            # Set response attributes
            span.set_attribute("gen_ai.completion", response.text)
            span.set_attribute("gen_ai.latency", duration_ms)
            if prompt_tokens:
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            if completion_tokens:
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            if prompt_tokens and completion_tokens:
                span.set_attribute("gen_ai.usage.total_tokens", prompt_tokens + completion_tokens)
            
            # Capture with session tracer
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                attrs = dict(current_span.attributes) if current_span.attributes else {}
                
                # Record in session tracer
                event = session_tracer.capture_llm_call(
                    system_id="gemini_agent",
                    system_state_before={"prompt": prompt},
                    system_state_after={"response": response.text}
                )
                
                if event:
                    print(f"✅ Captured Gemini call in session tracer")
                    print(f"   Provider detected: {event.metadata.get('provider', 'Unknown')}")
                    print(f"   Model: {event.model_name}")
                    print(f"   Duration: {event.latency_ms:.2f}ms" if event.latency_ms else "   Duration: N/A")
            
            print(f"\nResponse: {response.text}")
            print(f"Model: {model_name}")
            if prompt_tokens and completion_tokens:
                print(f"Usage - Prompt tokens: {prompt_tokens}")
                print(f"Usage - Completion tokens: {completion_tokens}")
                print(f"Usage - Total tokens: {prompt_tokens + completion_tokens}")
            
            # Save session trace
            filepath = session_tracer.end_session(save=True)
            if filepath:
                print(f"\n📁 OTEL session trace saved to: {filepath}")
            
            return {
                "provider": "google",
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (prompt_tokens + completion_tokens) if prompt_tokens and completion_tokens else None
                },
                "content": response.text,
                "duration_ms": duration_ms
            }
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            print(f"❌ Error with Gemini: {e}")
            # Save session trace even on error
            filepath = session_tracer.end_session(save=True)
            if filepath:
                print(f"\n📁 OTEL session trace saved to: {filepath}")
            return None


def test_gemini_streaming_with_otel():
    """Test Gemini streaming with OTEL tracing."""
    print("\n=== Testing Gemini Streaming with OTEL ===\n")
    
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    
    # Get OTEL tracer
    tracer = trace.get_tracer(__name__)
    
    # Create OTEL span
    with tracer.start_as_current_span("gemini.chat.completions.stream") as span:
        try:
            model_name = "gemini-1.5-flash"
            prompt = "Count to 5 slowly."
            
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.request.provider", "google")
            span.set_attribute("gen_ai.prompt", prompt)
            span.set_attribute("llm.streaming", True)
            
            start_time = time.time()
            first_token_time = None
            
            # Initialize model
            model = genai.GenerativeModel(model_name)
            
            # Make streaming call
            response = model.generate_content(prompt, stream=True)
            
            print("Response (streaming): ", end="")
            full_content = ""
            chunk_count = 0
            
            for chunk in response:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time()
                
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_content += chunk.text
            print()
            
            # Calculate timings
            total_duration_ms = (time.time() - start_time) * 1000
            time_to_first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else None
            
            # Set final attributes
            span.set_attribute("gen_ai.completion", full_content)
            span.set_attribute("gen_ai.latency", total_duration_ms)
            span.set_attribute("gen_ai.time_to_first_token", time_to_first_token_ms)
            span.set_attribute("gen_ai.chunk_count", chunk_count)
            
            print(f"\n✅ Gemini streaming traced with OTEL")
            print(f"   Duration: {total_duration_ms:.2f}ms")
            print(f"   Time to first token: {time_to_first_token_ms:.2f}ms" if time_to_first_token_ms else "   Time to first token: N/A")
            print(f"   Chunks: {chunk_count}")
            
            return {
                "provider": "google",
                "model": model_name,
                "streaming": True,
                "content": full_content,
                "duration_ms": total_duration_ms,
                "time_to_first_token_ms": time_to_first_token_ms,
                "chunk_count": chunk_count
            }
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            print(f"❌ Error with Gemini streaming: {e}")
            return None


def main():
    """Run OTEL-based tracing tests for Gemini."""
    print("🧪 Testing OTEL-Based Tracing for Gemini\n")
    
    # Check for Gemini API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
        print("Please set one of these in your .env file.")
        return
    
    print("✅ Gemini API key loaded")
    print()
    
    results = []
    
    # Test regular Gemini call
    result = test_gemini_with_otel()
    if result:
        results.append(result)
    
    # Test streaming
    result = test_gemini_streaming_with_otel()
    if result:
        results.append(result)
    
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print(f"Total traces collected: {len(results)}")
    print("\nOTEL-based tracing demonstrated for:")
    print("- ✅ Google Gemini (via OpenTelemetry manual instrumentation)")
    print("\nKey points:")
    print("- OTEL spans capture all LLM metadata")
    print("- SessionTracer can extract data from OTEL spans")
    print("- Works for any provider not directly supported by Langfuse")
    print("- Traces saved locally in otel_traces/ directory")


if __name__ == "__main__":
    main()