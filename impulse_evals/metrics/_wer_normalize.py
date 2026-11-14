# SPDX-License-Identifier: AGPL-3.0-only
"""Number, letter, date and ID normalisation before WER (FIXES-ROUND-2-SPEC.md A4 as revised by
"A4-rev"; METRICS.md section 2.1 VOICE-009 / VOICE-011).

``normalize_for_wer(text)`` makes the written and the spoken form of the same words compare equal:
``ORD-1004``, ``ORD 1004``, ``ORD1004`` and ``o r d dash one zero zero four`` all become ``ord 1004``;
``1990-04-12`` and ``twelfth April nineteen ninety`` both become ``19900412``. Steps, in order:

1. Characters (before punctuation is removed): NFKC; non-ASCII decimal digits mapped to ASCII; ISO
   and numeric dates become a date token (see step 4); thousands separators between digits are
   removed; a decimal point between digits is kept; ordinal suffixes after digits are removed (a
   digit-written ordinal word such as ``1st`` or ``21st`` is marked as an ordinal); ``%`` becomes
   `` percent ``; currency symbols become spaces; a single hyphen between exactly two digit groups of
   at most 3 digits each is a range boundary (``3-5``, ``10-12``), any other chain of hyphenated digit
   groups is written with spaces (``1800-123-4567`` -> ``1800 123 4567``, joined in step 5), and every
   other hyphen becomes a space in step 2; tokens are split at every letter/digit boundary
   (``ord1004`` -> ``ord 1004``).
2. Lowercase, apostrophes removed, other punctuation as spaces, split into tokens.
3. Number words to digits (cardinals, multipliers, ``and``, ordinals, years, ``point``, ``oh``).
   Arithmetic only on digit tokens of at most 15 digits (longer ones are kept as written); a zero is
   never multiplied (``0 lakh`` stays); ``oh`` is 0 only between two digit words or digit tokens, or
   right after a 10-99 year group (``nineteen oh five``); an ordinal ends a number. A digit-written
   ordinal (``1st``) is an ordinal like its word form.
4. Dates with a month name become a date token: year (4 digits, when present) + month + day (2
   digits each), in that fixed order. ``may`` / ``march`` / ``mar`` are a month only in
   ``<day> of <month>``, in ``<month> <day>`` with the day written as digits or as an ordinal word, or
   when a 4-digit year follows.
5. Joins: (a) a run of single letters with 3 or more letters, or 2 or more that touches a digit token
   or a spoken ``dash`` / ``hyphen``; (b) adjacent digit tokens, never across a range boundary or an
   ordinal; (c) a spoken ``dash`` / ``hyphen`` next to a digit token or a single letter (not the words
   ``a`` / ``i``) is dropped; (d) a single letter other than ``a`` / ``i``, directly before a
   letters-only token that is directly followed by a digit token, is joined to it.

Numeric dates: ISO ``yyyy-mm-dd`` and ``a/b/yyyy`` (or ``-``, ``.``) with one part > 12 (that part is
the day) have a known order. ``a/b/yyyy`` with both parts <= 12 is ambiguous: ``normalize_for_wer``
writes it as year + the two parts in written order; ``normalize_pair_for_wer`` (used by
``corpus_wer``) resolves it to the other side's known-order date token when that token has the same
year and the same two numbers.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import NamedTuple, Optional

__all__ = ["normalize_for_wer", "normalize_pair_for_wer"]

# Sentinels used inside the text of steps 1 and 2 only: Unicode private-use characters (category Co,
# so never punctuation); any copy in the input is replaced by a space first.
_DECIMAL = "\ue000"          # a decimal point between digits
_RANGE = "\ue001"            # a hyphen between two short digit groups (a token of its own)
_DATE = "\ue002"             # prefix of a date token whose day/month order is known
_AMBIGUOUS_DATE = "\ue003"   # prefix of a numeric date with both parts <= 12 (written order)
_ORDINAL_MARK = "\ue004"     # prefix of a digit-written ordinal ("1st", "21st") without its suffix
_SENTINELS = (_DECIMAL, _RANGE, _DATE, _AMBIGUOUS_DATE, _ORDINAL_MARK)

_APOSTROPHES = "'’ʼ"
_SPOKEN_DASHES = frozenset({"dash", "hyphen"})
_LETTER_WORDS = frozenset({"a", "i"})   # single letters that are also English words
_MAX_ARITHMETIC_DIGITS = 15


# ------------------------------------------------------------------------------------------------
# Tokens from step 3 on
# ------------------------------------------------------------------------------------------------


class _Token(NamedTuple):
    text: str
    kind: str


_WORD = "word"                         # anything that is not one of the kinds below
_DIGITS = "digits"                     # digits as written
_NUMBER = "number"                     # number words converted to digits (cardinal, year, decimal)
_ORDINAL = "ordinal"                   # an ordinal word converted to digits
_RANGE_BOUNDARY = "range"              # the step 1 range sentinel (dropped from the output)
_KNOWN_DATE = "date"                   # a date token in year + month + day order
_UNRESOLVED_DATE = "ambiguous_date"    # a numeric date with both parts <= 12, in written order


# ------------------------------------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_ORDINAL_UNITS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9,
}
_ORDINAL_TEENS = {
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
}
_ORDINAL_TENS = {
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
    "eightieth": 80, "ninetieth": 90,
}
_HUNDRED = "hundred"
_BIG = {
    "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
}
_ZERO = "zero"
_OH = "oh"
_AND = "and"
_POINT = "point"
_OF = "of"

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
# Month names that are also ordinary English words (A4-rev step 4).
_AMBIGUOUS_MONTHS = frozenset({"may", "march", "mar"})


class _Word(NamedTuple):
    kind: str       # unit | teen | tens | hundred | big
    value: int
    ordinal: bool


def _number_words() -> dict[str, _Word]:
    words: dict[str, _Word] = {}
    tables = (
        (_UNITS, "unit", False), (_TEENS, "teen", False), (_TENS, "tens", False),
        (_ORDINAL_UNITS, "unit", True), (_ORDINAL_TEENS, "teen", True), (_ORDINAL_TENS, "tens", True),
    )
    for table, kind, ordinal in tables:
        for word, value in table.items():
            words[word] = _Word(kind, value, ordinal)
    words[_HUNDRED] = _Word("hundred", 100, False)
    for word, value in _BIG.items():
        words[word] = _Word("big", value, False)
    return words


_WORDS = _number_words()
_OH_AS_ZERO = _Word("unit", 0, False)


# ------------------------------------------------------------------------------------------------
# Step 1: characters
# ------------------------------------------------------------------------------------------------

_DMY_DATE = re.compile(r"(?<![\d.])(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})(?![\d]|\.\d)")
_ISO_DATE = re.compile(r"(?<![\d.])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d]|\.\d)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
# A digit-written ordinal that is a word of its own ("1st", "21st"): marked as an ordinal (N-L1).
_DIGIT_ORDINAL = re.compile(rf"(?<!\w)(?<!{_DECIMAL})(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_ORDINAL_SUFFIX = re.compile(r"(?<=\d)(?:st|nd|rd|th)\b", re.IGNORECASE)
# Digit groups joined by hyphens or en dashes, with or without spaces ("3-5", "3 - 5", "1800-123-4567").
_HYPHEN = r"[ \t]*[-‐‑–][ \t]*"
_HYPHENATED_DIGITS = re.compile(rf"(?<!\d)\d+(?:{_HYPHEN}\d+)+")
_HYPHEN_SPLIT = re.compile(_HYPHEN)
_MAX_RANGE_GROUP_DIGITS = 3
_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def _ascii_digits(text: str) -> str:
    """Every non-ASCII decimal digit (e.g. Devanagari) replaced by its ASCII digit."""
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() and not char.isascii() else char for char in text
    )


def _known_date(year: str, month: int, day: int) -> str:
    return f" {_DATE}{year}{month:02d}{day:02d} "


def _iso_date(match: re.Match[str]) -> str:
    year, month, day = match.group(1), int(match.group(2)), int(match.group(3))
    if 1 <= month <= 12 and 1 <= day <= 31:
        return _known_date(year, month, day)
    return match.group(0)


def _dmy_date(match: re.Match[str]) -> str:
    """``a/b/yyyy``: a part > 12 is the day; both parts <= 12 is ambiguous (kept in written order)."""
    first, second, year = int(match.group(1)), int(match.group(3)), match.group(4)
    if not (1 <= first <= 31 and 1 <= second <= 31):
        return match.group(0)
    if first <= 12 and second <= 12:
        return f" {_AMBIGUOUS_DATE}{year}{first:02d}{second:02d} "
    if second <= 12:
        return _known_date(year, month=second, day=first)
    if first <= 12:
        return _known_date(year, month=first, day=second)
    return match.group(0)


def _hyphenated_digits(match: re.Match[str]) -> str:
    """A single hyphen between exactly two groups of at most 3 digits is a range boundary (``3-5``);
    any other hyphenated digit chain (``1800-123-4567``, ``98765-43210``) is written as spaces, so the
    groups join as one digit run in step 5 (N-M1)."""
    groups = _HYPHEN_SPLIT.split(match.group(0))
    if len(groups) == 2 and all(len(group) <= _MAX_RANGE_GROUP_DIGITS for group in groups):
        return f"{groups[0]} {_RANGE} {groups[1]}"
    return " ".join(groups)


def _step1(text: str) -> str:
    text = _ascii_digits(unicodedata.normalize("NFKC", text))
    for sentinel in _SENTINELS:
        text = text.replace(sentinel, " ")
    text = _ISO_DATE.sub(_iso_date, text)
    text = _DMY_DATE.sub(_dmy_date, text)
    text = _THOUSANDS.sub("", text)
    text = _DECIMAL_POINT.sub(_DECIMAL, text)
    text = _DIGIT_ORDINAL.sub(rf"{_ORDINAL_MARK}\1", text)
    text = _ORDINAL_SUFFIX.sub("", text)
    text = text.replace("%", " percent ")
    text = "".join(" " if unicodedata.category(char) == "Sc" else char for char in text)
    text = _HYPHENATED_DIGITS.sub(_hyphenated_digits, text)
    return _LETTER_DIGIT.sub(" ", text)


# ------------------------------------------------------------------------------------------------
# Step 2: lowercase, apostrophes, punctuation (as ``normalize_text``)
# ------------------------------------------------------------------------------------------------


def _step2(text: str) -> list[str]:
    out: list[str] = []
    for char in text.lower():
        if char in _APOSTROPHES:
            continue
        out.append(" " if unicodedata.category(char).startswith("P") else char)
    return [token.replace(_DECIMAL, ".") for token in "".join(out).split()]


def _classify(token: str) -> _Token:
    """The ``_Token`` of a step 2 token that step 3 kept as it is."""
    if token == _RANGE:
        return _Token(token, _RANGE_BOUNDARY)
    if token.startswith(_DATE):
        return _Token(token[1:], _KNOWN_DATE)
    if token.startswith(_AMBIGUOUS_DATE):
        return _Token(token[1:], _UNRESOLVED_DATE)
    if token.startswith(_ORDINAL_MARK):
        digits = token[1:]
        return _Token(digits, _ORDINAL if digits.isdecimal() else _WORD)
    return _Token(token, _DIGITS if token.isdecimal() else _WORD)


# ------------------------------------------------------------------------------------------------
# Step 3: number words to digits
# ------------------------------------------------------------------------------------------------


def _small_int(token: str) -> Optional[int]:
    """The value of a digit token short enough for arithmetic; None for any other token."""
    if token.isdecimal() and len(token) <= _MAX_ARITHMETIC_DIGITS:
        return int(token)
    return None


def _canonical_digits(token: str) -> str:
    """A digit token without leading zeros (no int conversion, so any length works)."""
    return token.lstrip("0") or "0"


def _is_digit(token: str) -> bool:
    """A cardinal digit word (``zero``..``nine``) or a digit token."""
    return token in _UNITS or token.isdecimal()


class _Numbers:
    """Left-to-right number-word parser over the step 2 tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.zero_oh = self._zero_ohs(tokens)

    @staticmethod
    def _zero_ohs(tokens: list[str]) -> set[int]:
        """Indices of ``oh`` tokens that mean 0: a run of ``oh`` between two digit words or digit
        tokens ("one oh four"). ``nineteen oh five`` is read by ``_year``."""
        zero: set[int] = set()
        start = 0
        while start < len(tokens):
            if tokens[start] != _OH:
                start += 1
                continue
            end = start
            while end < len(tokens) and tokens[end] == _OH:
                end += 1
            if start > 0 and end < len(tokens) and _is_digit(tokens[start - 1]) and _is_digit(tokens[end]):
                zero.update(range(start, end))
            start = end
        return zero

    def _word(self, index: int) -> Optional[_Word]:
        if index >= len(self.tokens):
            return None
        token = self.tokens[index]
        if token == _OH:
            return _OH_AS_ZERO if index in self.zero_oh else None
        return _WORDS.get(token)

    def _unit_value(self, index: int) -> Optional[int]:
        """The value of a cardinal unit word ``one``..``nine`` at ``index`` (not ``zero`` / ``oh``)."""
        word = self._word(index)
        if word is None or word.kind != "unit" or word.ordinal or word.value == 0:
            return None
        return word.value

    def _digit_word(self, index: int) -> Optional[int]:
        """The value of a cardinal digit word (``zero``..``nine``, or ``oh`` meaning 0)."""
        word = self._word(index)
        if word is None or word.kind != "unit" or word.ordinal:
            return None
        return word.value

    def _group(self, index: int) -> Optional[tuple[int, int]]:
        """A cardinal 10-99 group at ``index``: ``(value, next index)``."""
        word = self._word(index)
        if word is None or word.ordinal:
            return None
        if word.kind == "teen":
            return word.value, index + 1
        if word.kind == "tens":
            unit = self._unit_value(index + 1)
            if unit is not None:
                return word.value + unit, index + 2
            return word.value, index + 1
        return None

    def _year(self, index: int) -> Optional[tuple[str, int]]:
        """``nineteen ninety`` -> 1990, ``twenty twenty six`` -> 2026, ``nineteen oh five`` -> 1905."""
        first = self._group(index)
        if first is None:
            return None
        value, after = first
        if after < len(self.tokens) and self.tokens[after] == _OH:
            unit = self._unit_value(after + 1)
            return (str(value * 100 + unit), after + 2) if unit is not None else None
        second = self._group(after)
        if second is None:
            return None
        second_value, end = second
        following = self._word(end)
        if following is not None and following.kind in ("hundred", "big"):
            return None
        if end < len(self.tokens) and self.tokens[end] == _POINT:
            return None
        return str(value * 100 + second_value), end

    def _continues(self, index: int, last_multiplier: int) -> bool:
        """After ``and``: the next token continues the number (a word below a multiplier, or a
        digit token smaller than the last multiplier)."""
        if index >= len(self.tokens):
            return False
        word = self._word(index)
        if word is not None:
            return word.kind in ("unit", "teen", "tens") and word.value > 0
        value = _small_int(self.tokens[index])
        return value is not None and value < last_multiplier

    def _is_zero(self, index: int) -> bool:
        """``zero``, ``oh`` read as 0, or a digit token of zeros."""
        token = self.tokens[index]
        return token == _ZERO or index in self.zero_oh or (token.isdecimal() and not token.strip("0"))

    def _cardinal(self, index: int) -> Optional[tuple[str, int, bool]]:
        """A cardinal or ordinal number starting at ``index``: ``(digits, next index, ordinal)``."""
        tokens = self.tokens
        total = current = 0
        largest_big = last_multiplier = 0
        last: Optional[str] = None
        j = index
        start = tokens[index]
        if start.isdecimal():
            # A digit token followed by a multiplier word is multiplied ("10 lakh"); a zero never is.
            value = _small_int(start)
            follower = self._word(index + 1)
            if value is None or value == 0 or follower is None or follower.kind not in ("hundred", "big"):
                return None
            if follower.kind == "hundred" and value >= 100:
                return None
            current, last, j = value, "teen", index + 1
        else:
            word = self._word(index)
            if word is None:
                return None
            if word.kind in ("hundred", "big") and index > 0 and (
                tokens[index - 1].isdecimal() or self._is_zero(index - 1)
            ):
                # A multiplier right after a digit token it could not multiply, or after a zero,
                # stays a word ("0 lakh", "zero lakh").
                return None
        ended_by_ordinal = False
        while j < len(tokens):
            token = tokens[j]
            if token == _AND and last in ("hundred", "big") and self._continues(j + 1, last_multiplier):
                j += 1
                continue
            word = self._word(j)
            if word is None:
                value = _small_int(token)
                if value is not None and last in ("hundred", "big") and value < last_multiplier:
                    current += value
                    j += 1
                break
            if word.kind == "unit":
                if word.value == 0:
                    if last is not None:
                        break
                    current, last, j = 0, "zero", j + 1
                    break
                if last not in (None, "tens", "hundred", "big"):
                    break
            elif word.kind in ("teen", "tens"):
                if last not in (None, "hundred", "big"):
                    break
            elif word.kind == "hundred":
                if last not in (None, "unit", "teen", "tens") or current >= 100:
                    break
            else:  # big
                if (last == "big" and word.value <= largest_big) or last == "zero":
                    break
            j += 1
            if word.kind in ("unit", "teen", "tens"):
                current += word.value
                last = word.kind
            elif word.kind == "hundred":
                current = (current or 1) * 100
                last, last_multiplier = "hundred", 100
            else:
                part = current if current else (0 if last == "big" else 1)
                if word.value > largest_big and total:
                    total = (total + part) * word.value
                else:
                    total += part * word.value
                current, largest_big = 0, max(largest_big, word.value)
                last, last_multiplier = "big", word.value
            if word.ordinal:
                ended_by_ordinal = True
                break
        if last is None:
            return None
        number = total + current
        if not ended_by_ordinal:
            decimal = self._decimal(j)
            if decimal is not None:
                digits, j = decimal
                return f"{number}.{digits}", j, False
        return str(number), j, ended_by_ordinal

    def _decimal(self, index: int) -> Optional[tuple[str, int]]:
        """``point`` followed by digit words: ``(decimal digits, next index)``."""
        if index >= len(self.tokens) or self.tokens[index] != _POINT:
            return None
        digits: list[str] = []
        j = index + 1
        while j < len(self.tokens):
            value = self._digit_word(j)
            if value is None:
                break
            digits.append(str(value))
            j += 1
        if not digits:
            return None
        return "".join(digits), j

    def run(self) -> list[_Token]:
        out: list[_Token] = []
        tokens = self.tokens
        i = 0
        while i < len(tokens):
            year = self._year(i)
            if year is not None:
                out.append(_Token(year[0], _NUMBER))
                i = year[1]
                continue
            number = self._cardinal(i)
            if number is not None:
                digits, i, ordinal = number
                out.append(_Token(digits, _ORDINAL if ordinal else _NUMBER))
                continue
            if tokens[i].isdecimal():
                decimal = self._decimal(i + 1)
                if decimal is not None:
                    out.append(_Token(f"{_canonical_digits(tokens[i])}.{decimal[0]}", _NUMBER))
                    i = decimal[1]
                    continue
            out.append(_classify(tokens[i]))
            i += 1
        return out


# ------------------------------------------------------------------------------------------------
# Step 4: dates with a month name
# ------------------------------------------------------------------------------------------------


def _day(token: _Token) -> Optional[int]:
    if token.kind in (_DIGITS, _NUMBER, _ORDINAL) and token.text.isdecimal() and len(token.text) <= 2:
        value = int(token.text)
        return value if 1 <= value <= 31 else None
    return None


def _month(tokens: list[_Token], index: int) -> Optional[int]:
    if index < len(tokens) and tokens[index].kind == _WORD:
        return _MONTHS.get(tokens[index].text)
    return None


def _year_at(tokens: list[_Token], index: int) -> str:
    """The 4-digit year token at ``index``, or ``""``."""
    if index < len(tokens) and tokens[index].kind in (_DIGITS, _NUMBER):
        text = tokens[index].text
        if text.isdecimal() and len(text) == 4:
            return text
    return ""


def _date(year: str, month: int, day: int) -> _Token:
    return _Token(f"{year}{month:02d}{day:02d}", _KNOWN_DATE)


def _day_first_date(tokens: list[_Token], index: int) -> Optional[tuple[_Token, int]]:
    """``<day> [of] <month> [<year>]``; an ambiguous month word needs ``of`` or a year."""
    day = _day(tokens[index])
    if day is None:
        return None
    j = index + 1
    has_of = j < len(tokens) and tokens[j] == _Token(_OF, _WORD)
    if has_of:
        j += 1
    month = _month(tokens, j)
    if month is None:
        return None
    year = _year_at(tokens, j + 1)
    if tokens[j].text in _AMBIGUOUS_MONTHS and not has_of and not year:
        return None
    return _date(year, month, day), j + (2 if year else 1)


def _month_first_date(tokens: list[_Token], index: int) -> Optional[tuple[_Token, int]]:
    """``<month> <day> [<year>]``; for an ambiguous month word the day is written as digits or as an
    ordinal word, or a year follows."""
    month = _month(tokens, index)
    if month is None or index + 1 >= len(tokens):
        return None
    day = _day(tokens[index + 1])
    if day is None:
        return None
    year = _year_at(tokens, index + 2)
    if tokens[index].text in _AMBIGUOUS_MONTHS and tokens[index + 1].kind == _NUMBER and not year:
        return None
    return _date(year, month, day), index + (3 if year else 2)


def _step4(tokens: list[_Token]) -> list[_Token]:
    out: list[_Token] = []
    i = 0
    while i < len(tokens):
        date = _day_first_date(tokens, i) or _month_first_date(tokens, i)
        if date is not None:
            token, i = date
            out.append(token)
            continue
        out.append(tokens[i])
        i += 1
    return out


# ------------------------------------------------------------------------------------------------
# Step 5: joins
# ------------------------------------------------------------------------------------------------


def _single_letter(token: _Token) -> bool:
    return token.kind == _WORD and len(token.text) == 1 and token.text.isalpha()


def _letter_not_word(token: _Token) -> bool:
    """A single letter that is not also an English word (``a``, ``i``)."""
    return _single_letter(token) and token.text not in _LETTER_WORDS


def _is_digit_token(token: _Token) -> bool:
    return token.kind in (_DIGITS, _NUMBER, _ORDINAL) and token.text.isdecimal()


def _in_digit_run(token: _Token) -> bool:
    """A digit token that step 5b may concatenate (an ordinal never is)."""
    return token.kind in (_DIGITS, _NUMBER) and token.text.isdecimal()


def _is_spoken_dash(token: _Token) -> bool:
    return token.kind == _WORD and token.text in _SPOKEN_DASHES


def _run_end(tokens: list[_Token], start: int, member: Callable[[_Token], bool]) -> int:
    end = start
    while end < len(tokens) and member(tokens[end]):
        end += 1
    return end


def _neighbours(tokens: list[_Token], start: int, end: int) -> list[_Token]:
    """The tokens directly before ``start`` and at ``end`` (where they exist)."""
    return [tokens[index] for index in (start - 1, end) if 0 <= index < len(tokens)]


def _join_letters(tokens: list[_Token]) -> list[_Token]:
    """(a) A single-letter run of 3+ letters, or of 2+ letters next to a digit token or a spoken dash."""
    out: list[_Token] = []
    i = 0
    while i < len(tokens):
        end = _run_end(tokens, i, _single_letter)
        size = end - i
        if size >= 3 or (
            size == 2
            and any(_is_digit_token(token) or _is_spoken_dash(token) for token in _neighbours(tokens, i, end))
        ):
            out.append(_Token("".join(token.text for token in tokens[i:end]), _WORD))
            i = end
            continue
        out.append(tokens[i])
        i += 1
    return out


def _join_digits(tokens: list[_Token]) -> list[_Token]:
    """(b) Adjacent digit tokens are concatenated (a range boundary or an ordinal stops a run)."""
    out: list[_Token] = []
    i = 0
    while i < len(tokens):
        end = _run_end(tokens, i, _in_digit_run)
        if end - i >= 2:
            out.append(_Token("".join(token.text for token in tokens[i:end]), _DIGITS))
            i = end
            continue
        out.append(tokens[i])
        i += 1
    return out


def _drop_spoken_dashes(tokens: list[_Token]) -> list[_Token]:
    """(c) A spoken ``dash`` / ``hyphen`` next to a digit token or a single letter is dropped."""
    return [
        token
        for index, token in enumerate(tokens)
        if not (
            _is_spoken_dash(token)
            and any(_is_digit_token(other) or _letter_not_word(other) for other in _neighbours(tokens, index, index + 1))
        )
    ]


def _join_letter_prefixes(tokens: list[_Token]) -> list[_Token]:
    """(d) ``c ust 2001`` -> ``cust 2001``: a single letter (not ``a`` / ``i``) directly before a
    letters-only token that is directly followed by a digit token."""
    out: list[_Token] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (
            i + 2 < len(tokens)
            and _letter_not_word(token)
            and tokens[i + 1].kind == _WORD
            and tokens[i + 1].text.isalpha()
            and _is_digit_token(tokens[i + 2])
        ):
            out.append(_Token(token.text + tokens[i + 1].text, _WORD))
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _step5(tokens: list[_Token]) -> list[_Token]:
    tokens = _join_letters(tokens)
    tokens = _join_digits(tokens)
    tokens = _drop_spoken_dashes(tokens)
    tokens = _join_letter_prefixes(tokens)
    return [token for token in tokens if token.kind != _RANGE_BOUNDARY]


# ------------------------------------------------------------------------------------------------
# Public
# ------------------------------------------------------------------------------------------------


def _normalized_tokens(text: Optional[str]) -> list[_Token]:
    if not text:
        return []
    tokens = _Numbers(_step2(_step1(text))).run()
    return _step5(_step4(tokens))


def _resolve_dates(tokens: list[_Token], other: list[_Token]) -> str:
    """The words of ``tokens``, each ambiguous numeric date resolved against ``other``: written as the
    other side's known-order date token with the same year and the same two numbers, when there is
    one; otherwise in written order (two ambiguous dates are compared as written)."""
    known = {token.text for token in other if token.kind == _KNOWN_DATE}
    words: list[str] = []
    for token in tokens:
        text = token.text
        if token.kind == _UNRESOLVED_DATE:
            swapped = text[:4] + text[6:8] + text[4:6]
            if swapped in known:
                text = swapped
        words.append(text)
    return " ".join(words)


def normalize_for_wer(text: Optional[str]) -> str:
    """The text normalised for WER (FIXES-ROUND-2-SPEC.md A4 / A4-rev steps 1 to 5); ``""`` for no
    text. An ambiguous numeric date is written as year + the two parts in written order."""
    return " ".join(token.text for token in _normalized_tokens(text))


def normalize_pair_for_wer(reference: Optional[str], hypothesis: Optional[str]) -> tuple[str, str]:
    """``(reference, hypothesis)`` normalised for WER, with each side's ambiguous numeric dates
    resolved against the other side's known-order dates (A4-rev step 4; one pair at a time)."""
    reference_tokens = _normalized_tokens(reference)
    hypothesis_tokens = _normalized_tokens(hypothesis)
    return (
        _resolve_dates(reference_tokens, hypothesis_tokens),
        _resolve_dates(hypothesis_tokens, reference_tokens),
    )
