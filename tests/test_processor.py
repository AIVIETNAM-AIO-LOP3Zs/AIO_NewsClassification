"""
Tests for the English text pre-processing pipeline (TF-IDF path).
"""
import pytest

from app.ml.processor import TextProcessor


@pytest.fixture
def processor():
    return TextProcessor(remove_stopwords=True)


class TestTextCleaning:
    def test_lowercase(self, processor):
        assert processor.clean("BBC SPORT NEWS") == "bbc sport news"

    def test_removes_url(self, processor):
        result = processor.clean("Read more at https://www.bbc.com/sport/football")
        assert "http" not in result
        assert "bbc.com" not in result

    def test_removes_email(self, processor):
        result = processor.clean("Contact editor@bbc.co.uk for details")
        assert "@" not in result

    def test_removes_html(self, processor):
        result = processor.clean("<p>Arsenal <b>win</b> the league.</p>")
        assert "<" not in result

    def test_removes_contractions(self, processor):
        # "won't" → "won t" → "won" (t is 1 char, filtered)
        result = processor.clean("They won't stop playing")
        assert "'" not in result

    def test_collapses_whitespace(self, processor):
        result = processor.clean("BBC   Sport   Report")
        assert "  " not in result

    def test_removes_digits(self, processor):
        result = processor.clean("Arsenal won 3-0 on Sunday")
        assert "3" not in result
        assert "0" not in result


class TestStopwordRemoval:
    def test_removes_stopwords(self, processor):
        tokens = processor.tokenize("the football match was played at the stadium")
        assert "the" not in tokens
        assert "was" not in tokens
        assert "at" not in tokens

    def test_keeps_content_words(self, processor):
        tokens = processor.tokenize("football match stadium arsenal")
        assert "football" in tokens
        assert "arsenal" in tokens

    def test_single_char_filtered(self, processor):
        tokens = processor.tokenize("a b c football")
        assert "a" not in tokens
        assert "b" not in tokens

    def test_no_stopwords_mode(self):
        proc = TextProcessor(remove_stopwords=False)
        tokens = proc.tokenize("the football match was played")
        assert "the" in tokens
        assert "was" in tokens


class TestFullPipeline:
    def test_process_returns_string(self, processor):
        result = processor.process("Arsenal clinched the Premier League title on Sunday.")
        assert isinstance(result, str)

    def test_process_url_only_is_empty(self, processor):
        # URL removed → empty after cleaning
        result = processor.process("https://www.bbc.com/sport")
        assert result == ""

    def test_combined_text(self):
        from app.schemas.news import NewsClassifyRequest
        req = NewsClassifyRequest(
            headline="Arsenal win title",
            content="The club celebrated their victory."
        )
        assert req.combined_text().startswith("Arsenal win title.")
