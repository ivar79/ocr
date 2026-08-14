import editdistance

def character_error_rate(pred: str, target: str) -> float:
    """محاسبه CER — نرخ خطای کاراکتری"""
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return editdistance.eval(pred, target) / len(target)

def word_error_rate(pred: str, target: str) -> float:
    """محاسبه WER — نرخ خطای کلمه‌ای"""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return editdistance.eval(pred_words, target_words) / len(target_words)

def accuracy(pred: str, target: str) -> float:
    """دقت — درصد کاراکترهای صحیح"""
    cer = character_error_rate(pred, target)
    return max(0.0, 1.0 - cer)
