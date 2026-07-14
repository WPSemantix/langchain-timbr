"""Unit tests for the pure trigram matcher (langchain_timbr.trigram)."""

from langchain_timbr import trigram as tg


def test_snake_and_camel_split_into_same_grams():
    assert tg.to_trigram_set("product_return") == tg.to_trigram_set("ProductReturn")


def test_stopwords_dropped():
    grams = tg.to_trigram_set("the order of a customer")
    # 'the', 'of', 'a' are stopwords and contribute no grams
    assert grams == tg.to_trigram_set("order customer")


def test_short_word_kept_whole():
    assert "id" in tg.to_trigram_set("id")


def test_contains_matches_question_referencing_name():
    q = tg.to_trigram_set("show me all returned products")
    name = tg.to_trigram_set("product_return")
    assert tg.contains(name, q)


def test_contains_rejects_unrelated():
    q = tg.to_trigram_set("total shipment weight by region")
    name = tg.to_trigram_set("employee_salary")
    assert not tg.contains(name, q)


def test_tiny_name_requires_full_subset():
    q = tg.to_trigram_set("what is the id value")
    assert tg.contains(tg.to_trigram_set("id"), q)
    # a single shared gram is not enough for a tiny 2-gram name
    assert not tg.contains(tg.to_trigram_set("ab"), tg.to_trigram_set("abstract concept"))


def test_empty_inputs_are_false():
    assert not tg.contains(frozenset(), tg.to_trigram_set("anything"))
    assert not tg.contains(tg.to_trigram_set("order"), frozenset())


def test_floor_blocks_thin_overlap():
    # a large name sharing only two grams with the question stays below the floor
    q = tg.to_trigram_set("or")  # deliberately minimal
    name = tg.to_trigram_set("organization_hierarchy")
    assert not tg.contains(name, q, threshold=0.1, floor=3)


# --------------------------------------------------------------------------- #
# normalize / pad_question / to_tokens
# --------------------------------------------------------------------------- #
def test_normalize_collapses_non_alnum():
    assert tg.normalize("The LRT-line, schedule!") == "the lrt line schedule"


def test_pad_question_wraps_in_sentinels():
    assert tg.pad_question("Cost by region") == " cost by region "


def test_to_tokens_keeps_stopwords_and_splits():
    assert tg.to_tokens("HasMany_items") == ["has", "many", "items"]


# --------------------------------------------------------------------------- #
# matches(): trigram path for long names, whole-word fallback for short names
# --------------------------------------------------------------------------- #
def _match(name, question):
    q_padded = tg.pad_question(question)
    q_tri = tg.to_trigram_set(question)
    name_norm = " ".join(tg.to_tokens(name))
    name_tri = tg.to_trigram_set(name)
    return tg.matches(name_norm, name_tri, q_padded, q_tri)


def test_matches_long_name_uses_trigram_path():
    assert _match("product_return", "show me all returned products")
    assert not _match("employee_salary", "total shipment weight by region")


def test_matches_short_name_whole_word_hit():
    # 'lrt' -> single trigram -> whole-word fallback
    assert _match("lrt", "average lrt per station")
    assert _match("lrt", "the LRT-line schedule")


def test_matches_short_name_no_substring_false_positive():
    # 'lrt' must NOT match as a substring of 'filtration'
    assert not _match("lrt", "filtration systems")


def test_matches_four_char_name_whole_word():
    # 'cost' -> two trigrams (< floor) -> whole-word fallback
    assert _match("cost", "total cost by region")
    assert not _match("cost", "costume design review")


def test_matches_empty_name_is_false():
    assert not tg.matches("", frozenset(), tg.pad_question("anything"),
                          tg.to_trigram_set("anything"))

