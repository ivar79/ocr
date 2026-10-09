"""OCR metrics for evaluating text recognition quality."""


def _edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings using pure Python DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def character_error_rate(pred: str, target: str) -> float:
    """Calculate CER (Character Error Rate)."""
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return _edit_distance(pred, target) / len(target)


def word_error_rate(pred: str, target: str) -> float:
    """Calculate WER (Word Error Rate)."""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    # Compute edit distance on word lists
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n] / len(target_words)


def accuracy(pred: str, target: str) -> float:
    """Calculate accuracy (1 - CER)."""
    cer = character_error_rate(pred, target)
    return max(0.0, 1.0 - cer)
