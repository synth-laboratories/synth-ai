"""Unit tests for OpenAI SFT to ADAS converter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from synth_ai.products.graph_gepa.converters import (
    ConversionError,
    ConversionResult,
    convert_openai_sft,
    preview_conversion,
)
from synth_ai.products.graph_gepa.converters.openai_sft import (
    detect_system_prompt,
    extract_fields,
    infer_template,
    parse_sft_example,
    validate_sft_examples,
)


class TestParseSftExample:
    """Tests for parse_sft_example function."""

    def test_basic_parsing(self):
        """Test parsing a simple messages array."""
        example = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        system, user, assistant = parse_sft_example(example)
        assert system == "You are a helpful assistant."
        assert user == "What is 2+2?"
        assert assistant == "4"

    def test_no_system_prompt(self):
        """Test parsing without system prompt."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        system, user, assistant = parse_sft_example(example)
        assert system is None
        assert user == "Hello"
        assert assistant == "Hi there!"

    def test_multi_turn_uses_last_pair(self):
        """Test that multi-turn conversations use last user->assistant pair."""
        example = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Second answer"},
            ]
        }
        system, user, assistant = parse_sft_example(example)
        assert user == "Second question"
        assert assistant == "Second answer"

    def test_empty_messages(self):
        """Test parsing empty messages array."""
        example = {"messages": []}
        system, user, assistant = parse_sft_example(example)
        assert system is None
        assert user is None
        assert assistant is None


class TestDetectSystemPrompt:
    """Tests for detect_system_prompt function."""

    def test_all_same_system_prompt(self):
        """Test when all examples have the same system prompt."""
        examples = [
            {"messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
            {"messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ]
        result = detect_system_prompt(examples)
        assert result == "Be helpful."

    def test_most_common_system_prompt(self):
        """Test that most common system prompt is used."""
        examples = [
            {"messages": [{"role": "system", "content": "Common"}, {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
            {"messages": [{"role": "system", "content": "Common"}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
            {"messages": [{"role": "system", "content": "Rare"}, {"role": "user", "content": "Q3"}, {"role": "assistant", "content": "A3"}]},
        ]
        result = detect_system_prompt(examples)
        assert result == "Common"

    def test_no_system_prompts(self):
        """Test when no examples have system prompts."""
        examples = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ]
        result = detect_system_prompt(examples)
        assert result is None


class TestInferTemplate:
    """Tests for infer_template function."""

    def test_question_context_pattern(self):
        """Test detection of Question:/Context: pattern."""
        user_messages = [
            "Question: What is Python?\nContext: Python is a programming language.",
            "Question: What is JavaScript?\nContext: JavaScript runs in browsers.",
            "Question: What is Go?\nContext: Go is a compiled language.",
        ]
        template, field_names = infer_template(user_messages)
        assert template is not None
        assert "question" in field_names
        assert "context" in field_names

    def test_instruction_input_pattern(self):
        """Test detection of ### Instruction/### Input pattern."""
        user_messages = [
            "### Instruction\nTranslate to French\n### Input\nHello world",
            "### Instruction\nTranslate to French\n### Input\nGoodbye",
            "### Instruction\nTranslate to French\n### Input\nThank you",
        ]
        template, field_names = infer_template(user_messages)
        assert template is not None
        assert "instruction" in field_names
        assert "input" in field_names

    def test_no_pattern_detected(self):
        """Test fallback when no pattern detected."""
        user_messages = [
            "Random message one",
            "Another completely different format",
            "Third unique message style",
        ]
        template, field_names = infer_template(user_messages)
        assert template is None
        assert field_names == ["user_message"]

    def test_empty_messages(self):
        """Test handling of empty message list."""
        template, field_names = infer_template([])
        assert template is None
        assert field_names == ["user_message"]


class TestExtractFields:
    """Tests for extract_fields function."""

    def test_extract_question_context(self):
        """Test extracting question and context fields."""
        message = "Question: What is Python?\nContext: Python is a programming language."
        fields = extract_fields(message, ["question", "context"])
        assert fields["question"] == "What is Python?"
        assert fields["context"] == "Python is a programming language."

    def test_extract_instruction_input(self):
        """Test extracting instruction and input fields."""
        message = "### Instruction\nTranslate to French\n### Input\nHello world"
        fields = extract_fields(message, ["instruction", "input"])
        assert "Translate to French" in fields["instruction"]
        assert "Hello world" in fields["input"]

    def test_fallback_to_user_message(self):
        """Test fallback when no fields can be extracted."""
        message = "Just a plain message with no structure"
        fields = extract_fields(message, ["question", "context"])
        assert fields == {"user_message": message}


class TestValidateSftExamples:
    """Tests for validate_sft_examples function."""

    def test_valid_examples(self):
        """Test validation of valid examples."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        valid, warnings = validate_sft_examples(examples)
        assert len(valid) == 1
        assert len(warnings) == 0

    def test_missing_messages(self):
        """Test that missing 'messages' key is caught."""
        examples = [
            {"content": "no messages key"},  # Invalid - missing messages
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},  # Valid
        ]
        valid, warnings = validate_sft_examples(examples)
        assert len(valid) == 1
        assert len(warnings) == 1
        assert "Missing 'messages'" in warnings[0].message

    def test_missing_user_role(self):
        """Test that missing user role is caught."""
        examples = [
            {"messages": [{"role": "assistant", "content": "A"}]},  # Invalid - no user
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},  # Valid
        ]
        valid, warnings = validate_sft_examples(examples)
        assert len(valid) == 1
        assert len(warnings) == 1

    def test_missing_assistant_role(self):
        """Test that missing assistant role is caught."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}]},  # Invalid - no assistant
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},  # Valid
        ]
        valid, warnings = validate_sft_examples(examples)
        assert len(valid) == 1
        assert len(warnings) == 1

    def test_empty_assistant_skipped(self):
        """Test that empty assistant responses are skipped."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": ""}]},  # Invalid
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},  # Valid
        ]
        valid, warnings = validate_sft_examples(examples)
        assert len(valid) == 1
        assert len(warnings) == 1
        assert "Empty assistant" in warnings[0].message

    def test_no_valid_examples_raises(self):
        """Test that ConversionError is raised when no valid examples."""
        examples = [{"bad": "data"}]
        with pytest.raises(ConversionError, match="No valid examples found"):
            validate_sft_examples(examples)


class TestConvertOpenaiSft:
    """Tests for convert_openai_sft function."""

    def test_basic_conversion(self):
        """Test basic conversion from list of examples."""
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
        ]
        result = convert_openai_sft(examples)

        assert isinstance(result, ConversionResult)
        assert len(result.dataset["tasks"]) == 2
        assert len(result.dataset["gold_outputs"]) == 2
        assert result.dataset["metadata"]["task_description"] == "You are a helpful assistant."
        assert result.dataset["metadata"]["source_format"] == "openai_sft"

    def test_conversion_from_file(self):
        """Test conversion from JSONL file."""
        examples = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            temp_path = f.name

        try:
            result = convert_openai_sft(temp_path)
            assert len(result.dataset["tasks"]) == 2
        finally:
            Path(temp_path).unlink()

    def test_task_id_format(self):
        """Test that task IDs are formatted correctly."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        result = convert_openai_sft(examples)
        assert result.dataset["tasks"][0]["task_id"] == "sft_0000"

    def test_gold_output_structure(self):
        """Test gold output structure."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        result = convert_openai_sft(examples)
        gold = result.dataset["gold_outputs"][0]
        assert gold["task_id"] == "sft_0000"
        assert gold["output"]["response"] == "A"
        assert gold["score"] == 1.0

    def test_template_detection_disabled(self):
        """Test that template detection can be disabled."""
        examples = [
            {"messages": [{"role": "user", "content": "Question: What?\nContext: Something"}, {"role": "assistant", "content": "A"}]},
        ]
        result = convert_openai_sft(examples, detect_template=False)
        # Should use user_message, not extracted fields
        assert "user_message" in result.dataset["tasks"][0]["input"]

    def test_max_examples(self):
        """Test max_examples parameter."""
        examples = [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(10)
        ]
        result = convert_openai_sft(examples, max_examples=3)
        assert len(result.dataset["tasks"]) == 3

    def test_stats_populated(self):
        """Test that stats are populated correctly."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        result = convert_openai_sft(examples)
        assert "total_examples" in result.stats
        assert "output_examples" in result.stats
        assert "template_detected" in result.stats

    def test_no_examples_raises(self):
        """Test that empty input raises ConversionError."""
        with pytest.raises(ConversionError):
            convert_openai_sft([])


class TestPreviewConversion:
    """Tests for preview_conversion function."""

    def test_preview_limits_examples(self):
        """Test that preview limits number of examples."""
        examples = [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(10)
        ]
        preview = preview_conversion(examples, num_examples=2)
        assert len(preview["sample_tasks"]) == 2

    def test_preview_includes_metadata(self):
        """Test that preview includes metadata."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        preview = preview_conversion(examples)
        assert "metadata" in preview
        assert "stats" in preview


class TestIntegration:
    """Integration tests for the converter."""

    def test_converted_dataset_has_required_fields(self):
        """Test that output has all required ADAS fields."""
        examples = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        ]
        result = convert_openai_sft(examples)
        dataset = result.dataset

        # Check top-level keys
        assert "tasks" in dataset
        assert "gold_outputs" in dataset
        assert "metadata" in dataset

        # Check task structure
        task = dataset["tasks"][0]
        assert "task_id" in task
        assert "input" in task

        # Check gold_output structure
        gold = dataset["gold_outputs"][0]
        assert "task_id" in gold
        assert "output" in gold
        assert "score" in gold

        # Check metadata
        assert "name" in dataset["metadata"]
        assert "source_format" in dataset["metadata"]

    def test_template_detection_with_question_context(self):
        """Test full workflow with Question:/Context: template."""
        examples = [
            {
                "messages": [
                    {"role": "system", "content": "Answer questions concisely."},
                    {"role": "user", "content": "Question: What is Python?\nContext: Python is a programming language."},
                    {"role": "assistant", "content": "Python is a high-level programming language."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "Answer questions concisely."},
                    {"role": "user", "content": "Question: What is JavaScript?\nContext: JavaScript runs in browsers."},
                    {"role": "assistant", "content": "JavaScript is a scripting language for web development."},
                ]
            },
        ]
        result = convert_openai_sft(examples)

        # Check that template was detected
        assert result.stats["template_detected"]
        assert "question" in result.stats["detected_fields"]
        assert "context" in result.stats["detected_fields"]

        # Check that fields were extracted
        task_input = result.dataset["tasks"][0]["input"]
        assert "question" in task_input or "user_message" in task_input
