from memmachine.semantic_memory.category_tags import (
    CATEGORY_TAG_POLICIES,
    get_allowed_tags_for_category,
    normalize_category_tag,
)


def test_closed_category_policies_use_prompt_tag_keys():
    profile_tags = get_allowed_tags_for_category("profile")
    life_tags = get_allowed_tags_for_category("profile_life_context")
    personality_tags = get_allowed_tags_for_category("agent_personality")

    assert profile_tags is not None
    assert "others" in profile_tags
    assert "accounts" in profile_tags
    assert life_tags is not None
    assert "general_preference" in life_tags
    assert personality_tags is not None
    assert personality_tags == {"tone", "persona", "style", "boundaries"}


def test_default_tags_are_members_of_allowed_sets():
    for policy in CATEGORY_TAG_POLICIES.values():
        assert policy.default_tag in policy.allowed_tags


def test_normalize_category_tag_lowercases_and_remaps_unknown():
    assert normalize_category_tag("agent_personality", " Tone ") == "tone"
    assert normalize_category_tag("agent_personality", "not_a_real_tag") == "style"
    assert normalize_category_tag("profile", "ACCOUNTS") == "accounts"
    assert normalize_category_tag("profile", "weird") == "others"
    assert (
        normalize_category_tag("profile_life_context", "Life_Situation")
        == "life_situation"
    )
    assert normalize_category_tag("profile_life_context", "nope") == "interests"


def test_normalize_category_tag_passthrough_for_open_categories():
    assert normalize_category_tag("crm", " CustomTag ") == "customtag"
    assert get_allowed_tags_for_category("crm") is None
