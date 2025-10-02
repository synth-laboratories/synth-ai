def test_version_has_dev_suffix():
    import synth_ai
    assert hasattr(synth_ai, "__version__")
    assert ".dev" in synth_ai.__version__ or "-dev" in synth_ai.__version__



