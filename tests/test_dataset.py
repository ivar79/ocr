from utils.dataset import CharsetManager

def test_charset_manager():
    manager = CharsetManager()
    assert manager.blank_token == "[BLANK]"
    assert manager.char_to_idx["[BLANK]"] == 0
    assert "ا" in manager.char_to_idx
    assert manager.num_classes > 50
