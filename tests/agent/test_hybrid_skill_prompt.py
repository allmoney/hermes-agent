from pathlib import Path

import pytest
import yaml

from agent import prompt_builder


def _skill(home: Path, name: str, description: str, *, prompt_category: str | None = None) -> None:
    path = home / "skills" / "demo" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    category_line = f"prompt_category: {prompt_category}\n" if prompt_category else ""
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{category_line}---\n# {name}\nbody\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield home
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_hybrid_catalog_keeps_selected_descriptions_and_all_names(isolated_home):
    _skill(isolated_home, "alpha", "alpha routing description")
    _skill(isolated_home, "beta", "beta routing description")
    _skill(isolated_home, "gamma", "gamma routing description")
    (isolated_home / "config.yaml").write_text(
        yaml.safe_dump({
            "skills": {
                "prompt_catalog": {
                    "mode": "hybrid",
                    "described_names": ["alpha"],
                }
            }
        }),
        encoding="utf-8",
    )

    rendered = prompt_builder.build_skills_system_prompt()

    assert "- alpha: alpha routing description" in rendered
    assert "demo [names only]: beta, gamma" in rendered
    assert "beta routing description" not in rendered
    assert "gamma routing description" not in rendered
    assert "Every skill name remains visible" in rendered


def test_full_catalog_remains_backward_compatible_by_default(isolated_home):
    _skill(isolated_home, "alpha", "alpha routing description")
    _skill(isolated_home, "beta", "beta routing description")

    rendered = prompt_builder.build_skills_system_prompt()

    assert "- alpha: alpha routing description" in rendered
    assert "- beta: beta routing description" in rendered
    assert "[names only]" not in rendered


def test_skill_policy_requires_minimal_sufficient_set_not_partial_matches(isolated_home):
    _skill(isolated_home, "alpha", "alpha routing description")

    rendered = prompt_builder.build_skills_system_prompt()

    assert "minimal sufficient set" in rendered
    assert "do not load skills merely because they are partially related" in rendered


def test_prompt_category_groups_without_moving_skill_files(isolated_home):
    _skill(
        isolated_home,
        "cron-debug",
        "Use when cron jobs fail",
        prompt_category="devops/cron",
    )

    rendered = prompt_builder.build_skills_system_prompt()

    assert "  devops/cron:" in rendered
    assert "- cron-debug: Use when cron jobs fail" in rendered
    assert "  demo:" not in rendered
