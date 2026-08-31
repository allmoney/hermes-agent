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


def _scoped_skill(home: Path, name: str, description: str) -> None:
    path = home / "skills" / "demo" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump({
        "name": name,
        "description": description,
        "metadata": {"hermes": {"offer_scope": {
            "projects": ["family-budget-bot"],
            "platforms": ["telegram"],
            "toolsets": ["terminal"],
        }}},
    }, sort_keys=False)
    path.write_text(f"---\n{frontmatter}---\n# body\n", encoding="utf-8")


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


def test_offer_scope_demotes_description_but_keeps_name(isolated_home, monkeypatch):
    _scoped_skill(isolated_home, "budget-prod", "family budget production workflow")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    rendered = prompt_builder.build_skills_system_prompt(
        available_toolsets={"web"}, project_hints={"other-project"}
    )
    assert "budget-prod" in rendered
    assert "family budget production workflow" not in rendered


def test_offer_scope_keeps_description_when_all_hints_match(isolated_home, monkeypatch):
    _scoped_skill(isolated_home, "budget-prod", "family budget production workflow")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    rendered = prompt_builder.build_skills_system_prompt(
        available_toolsets={"terminal", "file"}, project_hints={"family-budget-bot"}
    )
    assert "- budget-prod: family budget production workflow" in rendered


def test_offer_scope_is_fail_open_when_project_and_platform_unknown(isolated_home, monkeypatch):
    _scoped_skill(isolated_home, "budget-prod", "family budget production workflow")
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    rendered = prompt_builder.build_skills_system_prompt(
        available_toolsets={"terminal"}, project_hints=set()
    )
    assert "- budget-prod: family budget production workflow" in rendered
