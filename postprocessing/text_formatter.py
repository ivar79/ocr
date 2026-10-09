import re


class TextPostProcessor:
    """Text postprocessor for Persian OCR output."""

    def __init__(self):
        self.char_corrections = {
            '\u0643': '\u06a9',  # Arabic kaf -> Persian kaf
            '\u064a': '\u06cc',  # Arabic yeh -> Persian yeh
            '\u0629': '\u0647',  # Arabic teh marbuta -> Persian heh
            '\u0623': '\u0627',  # alef with hamza above -> alef
            '\u0625': '\u0627',  # alef with hamza below -> alef
            '\u0649': '\u06cc',  # alef maksura -> Persian yeh
            '\u06c0': '\u0647',  # heh with yeh above -> heh
            '\u06c2': '\u0647',  # heh goal -> heh
            '\u06d4': '\u060c',  # arabic full stop -> arabic comma
        }
        # Arabic-Indic digits (٠-٩) -> Persian digits (۰-۹)
        self.arabic_to_persian_digits = str.maketrans(
            '\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669',
            '\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9',
        )
        # High-frequency Persian words used for conservative, exact-match
        # correction of common single-letter OCR substitutions.
        self.frequent_words = {
            'از', 'به', 'با', 'در', 'برای', 'که', 'این', 'آن', 'یک', 'دو',
            'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده', 'و', 'را',
            'ما', 'شما', 'من', 'تو', 'او', 'آنها', 'مردم', 'هر', 'همه',
            'هیچ', 'برخی', 'بعضی', 'چند', 'سال', 'ماه', 'روز', 'هفته',
            'ساعت', 'دقیقه', 'کار', 'خانه', 'مدرسه', 'دانشگاه', 'کتاب',
            'نامه', 'عکس', 'تصویر', 'متن', 'زبان', 'فارسی', 'انگلیسی',
            'کشور', 'شهر', 'خیابان', 'درخت', 'گل', 'آب', 'نان', 'غذا',
            'چای', 'قهوه', 'شیر', 'میوه', 'سیب', 'ماشین', 'اتوبوس',
            'قطار', 'هواپیما', 'پول', 'قیمت', 'بازار', 'مغازه', 'خرید',
            'فروش', 'مشتری', 'فروشنده', 'کالا', 'نوع', 'اندازه', 'رنگ',
            'سیاه', 'سفید', 'قرمز', 'آبی', 'سبز', 'زرد', 'بزرگ', 'کوچک',
            'بلند', 'کوتاه', 'زیاد', 'کم', 'خوب', 'بد', 'عالی', 'زیبا',
            'خوشحال', 'خسته', 'سریع', 'آسان', 'سخت', 'مهم', 'لازم',
            'ممکن', 'باید', 'شاید', 'هست', 'نیست', 'بود', 'باشد',
            'کرده', 'میکند', 'میکنم', 'رفت', 'آمد', 'خورد', 'نوشید',
            'خواند', 'نوشت', 'دید', 'شنید', 'گفت', 'خواست', 'گرفت',
            'داد', 'ساخت', 'جدید', 'قدیم', 'امروز', 'دیروز', 'فردا',
            'امسال', 'پارسال', 'بعد', 'قبل', 'حالا', 'اکنون', 'پس',
            'درباره', 'بدون', 'بالا', 'پایین', 'داخل', 'خارج', 'وسط',
            'کنار', 'روی', 'زیر', 'چرا', 'چگونه', 'کجا', 'کی', 'چه',
            'سلام', 'دنیا', 'لطفا', 'متشکرم', 'ببخشید', 'خواهش', 'بله',
            'خیر', 'باشه', 'اینجا', 'آنجا', 'همین', 'همان', 'خودش',
            'خودم', 'چیزی', 'کسی', 'جایی', 'زمان', 'مکان', 'انسان',
            'دوست', 'خانواده', 'پدر', 'مادر', 'برادر', 'خواهر',
            'پسر', 'دختر', 'مرد', 'زن', 'کودک', 'بچه', 'دست', 'پا',
            'سر', 'چشم', 'گوش', 'دهان', 'قلب', 'دستور', 'سوال',
            'جواب', 'سخن', 'کلام', 'صدای', 'نور', 'آتش', 'باد',
            'ابر', 'باران', 'برف', 'کوه', 'دریا', 'صحرا', 'جنگل',
        }
        self._word_cache = {}
        self._word_set = set(self.frequent_words)

    def process(self, text):
        if not text:
            return ""
        text = self._remove_gibberish(text)
        text = self._fix_common_errors(text)
        text = self._normalize_persian(text)
        text = self._normalize_digits(text)
        text = self._fix_punctuation(text)
        text = self._fix_words(text)
        text = self._fix_spacing(text)
        return text

    def _remove_gibberish(self, text):
        # Allowed punctuation stays; anything else is stripped
        allowed = r'\w\s.,;:!?()\-\u060C\u061B\u061F\u00AB\u00BB\u200C\u200D'
        text = re.sub(rf'[^{allowed}]', ' ', text)
        # Remove repeated non-punctuation gibberish (e.g. ~~~, ###) but keep
        # repeated punctuation for _fix_punctuation to collapse (e.g. !!! -> !)
        text = re.sub(rf'([^{allowed}])\1{{2,}}', ' ', text)
        return text.strip()

    def _fix_common_errors(self, text):
        for wrong, correct in self.char_corrections.items():
            text = text.replace(wrong, correct)
        return text

    def _normalize_persian(self, text):
        text = text.replace('\u0643', '\u06a9')
        text = text.replace('\u064a', '\u06cc')
        return text

    def _normalize_digits(self, text):
        """Convert Arabic-Indic digits (٠-٩) to Persian digits (۰-۹)."""
        return text.translate(self.arabic_to_persian_digits)

    def _fix_punctuation(self, text):
        """Collapse repeated punctuation."""
        text = re.sub(r'([.,;:!?،؛؟])\1+', r'\1', text)
        return text

    def _fix_words(self, text):
        """Conservative single-substitution word corrections for Persian.

        A token is replaced only when it is exactly one edit away (same
        length) from a unique high-frequency Persian word. Numbers, short
        tokens, and non-Persian tokens are never touched.
        """
        if not self.frequent_words:
            return text

        def correct_token(match):
            token = match.group(0)
            stripped = token.strip('.,;:!?()\'"«»،؛؟')
            if len(stripped) < 3 or len(stripped) > 12:
                return token
            if not self._is_persian(stripped):
                return token
            if stripped.isdigit():
                return token
            # Never modify a word that is already correct
            if stripped in self._word_set:
                return token
            candidate = self._closest_word(stripped)
            if candidate is None:
                return token
            return token.replace(stripped, candidate)

        return re.sub(r'\S+', correct_token, text)

    def _closest_word(self, token):
        """Return the unique high-frequency word at edit distance 1 (same length)."""
        if token in self._word_cache:
            return self._word_cache[token]
        candidates = []
        for word in self._word_set:
            if len(word) != len(token):
                continue
            if self._edit_distance(token, word) == 1:
                candidates.append(word)
                if len(candidates) > 1:
                    break
        result = candidates[0] if len(candidates) == 1 else None
        self._word_cache[token] = result
        return result

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein distance between two strings."""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]

    @staticmethod
    def _is_persian(token: str) -> bool:
        return any(
            '\u0600' <= ch <= '\u06ff' or '\u0750' <= ch <= '\u077f'
            for ch in token
        )

    def _fix_spacing(self, text):
        """Normalize whitespace while preserving line structure."""
        text = re.sub(r'[ \t]+', ' ', text)
        # Trim spaces around newlines, then collapse consecutive newlines
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'\n{2,}', '\n', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?،؛؟])', r'\1', text)
        return text.strip()
