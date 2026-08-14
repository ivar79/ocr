from inference import OCREngine

def test_engine_init():
    engine = OCREngine()
    assert engine.device is not None
