from postprocessing.text_formatter import TextPostProcessor

def test_persian_normalization():
    processor = TextPostProcessor()
    # Arabic kaf and yeh
    text = "كتابي"
    assert processor.process(text) == "کتابی"

def test_spacing():
    processor = TextPostProcessor()
    text = "سلام   دنیا ."
    assert processor.process(text) == "سلام دنیا."
