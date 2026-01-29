from synth_ai.sdk.optimization.internal.builders import _normalize_mipro_section
from synth_ai.sdk.optimization.internal.configs.prompt_learning import PromptLearningConfig


def _make_mipro_config() -> PromptLearningConfig:
    return PromptLearningConfig.from_mapping(
        {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://example.com",
                "mipro": {},
            }
        }
    )


def test_normalize_mipro_section_promotes_top_level_fields() -> None:
    pl_cfg = _make_mipro_config()
    config_dict = {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_url": "http://example.com",
            "bootstrap_train_seeds": [1, 2],
            "online_pool": [3],
            "test_pool": [4],
            "reference_pool": [5],
        }
    }

    _normalize_mipro_section(pl_cfg, config_dict, source="test", prefer_model=True)

    mipro_section = config_dict["prompt_learning"]["mipro"]
    assert mipro_section["bootstrap_train_seeds"] == [1, 2]
    assert mipro_section["online_pool"] == [3]
    assert mipro_section["test_pool"] == [4]
    assert mipro_section["reference_pool"] == [5]


def test_normalize_mipro_section_sets_env_name() -> None:
    pl_cfg = _make_mipro_config()
    config_dict = {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_url": "http://example.com",
            "mipro": {
                "env_name": "banking77",
                "bootstrap_train_seeds": [1],
                "online_pool": [2],
            },
        }
    }

    _normalize_mipro_section(pl_cfg, config_dict, source="test", prefer_model=True)

    pl_section = config_dict["prompt_learning"]
    assert pl_section["env_name"] == "banking77"
