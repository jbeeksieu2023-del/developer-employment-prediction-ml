import pandas as pd

from feature_engineering import (
    BinarySkillTransformer,
    GroupedSkillTransformer,
    SkillCountTransformer,
)
from process_skills import SkillVocabulary, parse_skills


def test_parse_skills_normalises_and_deduplicates_tokens():
    values = pd.Series([" Python ; SQL;python ", None, ""])

    assert parse_skills(values) == [["python", "sql"], [], []]


def test_vocabulary_is_learned_from_training_data_only():
    vocabulary = SkillVocabulary(min_freq=1).fit([["python"], ["sql"]])

    assert vocabulary.transform([["python", "rust"]]) == [[0]]


def test_skill_transformers_return_expected_shapes():
    values = pd.Series(["Python;SQL", "JavaScript;HTML/CSS", ""])

    binary = BinarySkillTransformer(min_freq=1).fit_transform(values)
    counts = SkillCountTransformer().fit_transform(values)
    grouped = GroupedSkillTransformer(mode="both").fit_transform(values)

    assert binary.shape == (3, 4)
    assert counts["total_skills"].tolist() == [2, 2, 0]
    assert grouped.shape == (3, 16)
