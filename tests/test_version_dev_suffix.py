def test_version_string_is_present():
    import synth_ai

    version = getattr(synth_ai, "__version__", None)

    assert isinstance(version, str), "__version__ should be a string"
    assert version.strip(), "__version__ should not be empty"


