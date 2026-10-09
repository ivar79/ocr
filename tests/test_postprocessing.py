from postprocessing.text_formatter import TextPostProcessor


class TestTextPostProcessor:
    """Tests for TextPostProcessor class."""

    def test_arabic_kaf_to_persian(self):
        """Arabic kaf should be converted to Persian kaf."""
        processor = TextPostProcessor()
        assert processor.process("كتاب") == "کتاب"

    def test_arabic_yeh_to_persian(self):
        """Arabic yeh should be converted to Persian yeh."""
        processor = TextPostProcessor()
        assert processor.process("مي") == "می"

    def test_arabic_teh_marbuta(self):
        """Arabic teh marbuta should be converted to Persian heh."""
        processor = TextPostProcessor()
        assert processor.process("ة") == "ه"

    def test_multiple_arabic_chars(self):
        """Multiple Arabic chars should all be converted."""
        processor = TextPostProcessor()
        result = processor.process("كتابي")
        assert result == "کتابی"

    def test_spacing_normalization(self):
        """Multiple spaces should be normalized to single space."""
        processor = TextPostProcessor()
        assert processor.process("سلام   دنیا") == "سلام دنیا"

    def test_punctuation_spacing(self):
        """Space before punctuation should be removed."""
        processor = TextPostProcessor()
        result = processor.process("سلام .")
        assert result == "سلام."

    def test_arabic_digits_to_persian(self):
        """Arabic-Indic digits should be converted to Persian digits."""
        processor = TextPostProcessor()
        assert processor.process("۱۲٣٤") == "۱۲۳۴"

    def test_alef_hamza_normalization(self):
        """Alef with hamza should normalize to plain alef (yeh -> Persian yeh)."""
        processor = TextPostProcessor()
        assert processor.process("أمل") == "امل"
        assert processor.process("إيران") == "ایران"

    def test_punctuation_dedupe(self):
        """Repeated punctuation should collapse to a single mark."""
        processor = TextPostProcessor()
        assert processor.process("سلام!!!") == "سلام!"

    def test_zwnj_preserved(self):
        """ZWNJ (نیم‌فاصله) should be preserved, not turned into a space."""
        processor = TextPostProcessor()
        assert processor.process("می\u200cروم") == "می\u200cروم"

    def test_word_correction_conservative(self):
        """Word corrections only fire for unique distance-1 Persian tokens."""
        processor = TextPostProcessor()
        # 'ساام' -> 'سلام' (single same-length substitution, unique candidate)
        assert processor.process("ساام دنیا") == "سلام دنیا"
        # Correct dictionary words, English tokens, and numbers are untouched
        assert processor.process("hello world") == "hello world"
        assert processor.process("سلام") == "سلام"
        assert processor.process("قیمت 123") == "قیمت 123"

    def test_whitespace_only(self):
        """Whitespace-only string should return empty string."""
        processor = TextPostProcessor()
        assert processor.process("   ") == ""

    def test_gibberish_removal(self):
        """Gibberish characters should be removed."""
        processor = TextPostProcessor()
        result = processor.process("hello ~^~ world")
        # Gibberish chars should be removed
        assert "~" not in result
        assert "^" not in result

    def test_preserve_normal_text(self):
        """Normal text should be preserved."""
        processor = TextPostProcessor()
        assert processor.process("Hello World") == "Hello World"

    def test_preserve_persian_text(self):
        """Normal Persian text should be preserved."""
        processor = TextPostProcessor()
        assert processor.process("سلام دنیا") == "سلام دنیا"

    def test_mixed_persian_english(self):
        """Mixed Persian and English should be preserved."""
        processor = TextPostProcessor()
        result = processor.process("Hello سلام World دنیا")
        assert "Hello" in result
        assert "سلام" in result
        assert "World" in result
        assert "دنیا" in result

    def test_newline_normalization(self):
        """Single newlines are preserved (line structure matters for OCR)."""
        processor = TextPostProcessor()
        result = processor.process("line1\nline2")
        assert "\n" in result

    def test_double_newline_normalization(self):
        """Multiple consecutive newlines collapse to a single newline."""
        processor = TextPostProcessor()
        result = processor.process("line1\n\nline2")
        assert result == "line1\nline2"
