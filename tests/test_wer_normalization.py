# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 2, A4 (DEFECT-3) as revised by A4-rev: number, letter, date and ID normalisation
before WER (FIXES-ROUND-2-SPEC.md A4 and "A4-rev"; METRICS.md section 2.1 VOICE-009 / VOICE-011).

``metrics/voice.py`` has ``normalize_for_wer(text) -> str``, used on both sides by ``corpus_wer``
(so VOICE-009, VOICE-011 and the run warning "Harness round-trip WER is high; ..."). ``normalize_text``
keeps its behaviour for VOICE-013 and its other callers.

A4-rev (after Stage D review R2) SUPERSEDES the A4 steps where they differ: IDs are canonicalised by
SPLITTING at letter/digit boundaries (``ORD-1004`` -> ``ord 1004``, no join marks); dates use the
fixed order year + month + day whenever the order is known, and an ambiguous numeric date (both
parts <= 12) is resolved per pair by ``corpus_wer`` against the other side's known-order date with
the same year and the same two numbers, otherwise written as year + the two parts in written order;
ranges and ordinals stop digit-run concatenation; ``a`` / ``i`` and ``may`` / ``march`` / ``oh`` no
longer change plain sentences; non-ASCII decimal digits map to ASCII; ``0 lakh`` stays; very long
digit tokens never raise.

Pinned here:
* every expected pair of A4-rev (``corpus_wer``), the mishearings A4-rev must still count, and the
  no-regression set from review R2 (WER never above the old ``normalize_text`` WER);
* ``normalize_for_wer`` outputs implied by the A4-rev rules (steps 1 to 5); the standalone output of
  an ambiguous numeric date is the written order (no other side to resolve against);
* the accepted residuals A4-rev lists (``CU ST dash 2001``: 2 word errors; currency symbols dropped;
  a comma does not stop a digit run; ``ten thirty`` read as 1030);
* run level, recomputed from the campaign transcripts (trimmed copies in
  ``tests/data/fixes_round2/wer/``: turns, ``heard_text`` and ``metadata["impulse_voice"]`` records
  without word timings): no VOICE-011 is poor in BankBot voice runs 20261023T200424Z-3c1d0d (normal)
  and 20261023T202802Z-088f33 (BUGGY), so the runner's round-trip warning condition (any counted
  VOICE-011 labelled poor) is not met; VOICE-009 is good in SupportBot voice run
  20261021T201818Z-cd156f ``order_status``, ``refund_within_policy``, ``damaged_item_ticket``;
* an end-to-end voice run (mock voice server) whose round trip hears "C UST dash 2001" no longer
  produces the run warning.

Re-check revisions (FIXES-ROUND-2-SPEC.md "Re-check revisions"; tests at the end of this file):

* N-M1: a hyphen between digit groups is a range boundary only when it is the single hyphen between
  exactly two groups of at most 3 digits (``3-5``, ``10-12``); a chain of hyphenated groups, or a group
  longer than 3 digits, joins like spaces (phone, Aadhaar, card and toll-free numbers);
* N-L1: a digit-written ordinal (``1st``, ``2nd``, ``21st``) is an ordinal like its word form;
* R2-L5: the sentinel characters in ``metrics/_wer_normalize.py`` are written as escapes, never as raw
  private-use characters (U+E000..U+F8FF).

Every run-level expectation was re-checked by hand against the A4-rev rules for every pair involved
(the pairs whose raw texts differ only in sentence punctuation normalise identically on both sides).
Under A4-rev the remaining round-trip errors are the residual "CU ST dash 2001" / "CU S T dash 2001" /
"CU ST dash 2 001" (2 errors each: ``cu st 2001`` vs ``cust 2001``) and two place-name mishearings
("Koregaon" heard as "Khoragon" / "Kuragan"), at most 4 errors in about 84 reference words per
scenario (well under 0.10), so no scenario is poor. The genuine mishearing "I lost by debit card" is
in a VOICE-009 heard_text (BankBot normal ``lost_card``), not in a round trip, so it does not touch
VOICE-011; no run-level VOICE-009 expectation is pinned for BankBot.
"""

import itertools
import json
from pathlib import Path

import pytest
from conftest import pcm_tone, wav_bytes

DATA = Path(__file__).parent / "data" / "fixes_round2" / "wer"
BANKBOT_NORMAL = "20261023T200424Z-3c1d0d"
BANKBOT_BUGGY = "20261023T202802Z-088f33"
SUPPORTBOT_VOICE = "20261021T201818Z-cd156f"
ROUND_TRIP_WARNING = "Harness round-trip WER is high; audio metrics in this run are unreliable."


def _normalize_for_wer(text):
    from impulse_evals.metrics.voice import normalize_for_wer

    return normalize_for_wer(text)


def _corpus_wer(pairs):
    from impulse_evals.metrics.voice import corpus_wer

    return corpus_wer(pairs)


def _old_wer(pairs):
    """The WER before fixes round 2: jiwer on ``normalize_text`` of both sides (the old behaviour)."""
    import jiwer

    from impulse_evals.metrics.voice import normalize_text

    references = [normalize_text(reference) for reference, _ in pairs]
    hypotheses = [normalize_text(hypothesis) for _, hypothesis in pairs]
    output = jiwer.process_words(references, hypotheses)
    errors = output.substitutions + output.deletions + output.insertions
    return errors / sum(len(reference.split()) for reference in references)


# =============================================================================================
# corpus_wer: the expected pairs of spec A4, as kept by A4-rev
# =============================================================================================

ZERO_WER_PAIRS = [
    ("My order ID is ORD-1004.", "My order ID is o r d dash one zero zero four."),
    ("My customer ID is CUST-2001.", "My customer ID is C UST dash 2001."),
    ("It is CUST-2001.", "It is CUST-two thousand 1."),
    ("My date of birth is twelfth April nineteen ninety.", "My date of birth is 04/12/1990."),
    (
        "I want to schedule an electricity bill payment of 2500 rupees on 2026-10-30.",
        "I want to schedule an electricity bill payment of 2,500 rupees on twenty twenty six ten thirty.",
    ),
    ("It ends with 4821.", "It ends with four thousand eight hundred and twenty one."),
    ("pin code 411001", "PIN code four one one zero zero one"),
    # A4-rev additions
    ("My date of birth is 1990-04-12.", "My date of birth is nineteen ninety zero four twelve."),
    ("on 2026-10-30", "on twenty twenty six ten thirty"),
]


@pytest.mark.parametrize("reference, hypothesis", ZERO_WER_PAIRS)
def test_spec_pairs_have_zero_wer(reference, hypothesis):
    assert _corpus_wer([(reference, hypothesis)]) == 0.0


@pytest.mark.parametrize("reference, hypothesis", ZERO_WER_PAIRS)
def test_spec_pairs_have_zero_wer_both_ways(reference, hypothesis):
    # The normalisation is applied to both sides, so swapping reference and hypothesis is also 0.
    assert _corpus_wer([(hypothesis, reference)]) == 0.0


def test_spec_pairs_as_one_corpus_have_zero_wer():
    assert _corpus_wer(ZERO_WER_PAIRS) == 0.0


def test_hyphenated_word_is_not_a_regression():
    # Passes before the fix too: "follow-up" and "follow up" are the same words.
    assert _corpus_wer([("follow-up", "follow up")]) == 0.0
    assert _corpus_wer([("Please follow-up tomorrow.", "please follow up tomorrow")]) == 0.0


def test_accepted_residual_cu_st_is_two_errors():
    # A4-rev: "CU ST dash 2001" normalises to "cu st 2001" (3 tokens), "CUST-2001" to "cust 2001"
    # (2 tokens): one substitution plus one insertion or deletion, i.e. 2 word errors either way round.
    assert _corpus_wer([("CUST-2001", "CU ST dash 2001")]) == pytest.approx(1.0)        # 2 errors / 2 words
    assert _corpus_wer([("CU ST dash 2001", "CUST-2001")]) == pytest.approx(2 / 3)      # 2 errors / 3 words


def test_real_mishearing_still_counts():
    # Regression guard: normalisation never hides a genuinely different word.
    assert _corpus_wer([("Hello, I lost my debit card.", "Hello. I lost by debit card.")]) == pytest.approx(1 / 6)


def test_corpus_wer_without_reference_words_is_none():
    # Regression guard (passes before the fix too).
    assert _corpus_wer([("", "anything"), (None, None)]) is None


# =============================================================================================
# A4-rev: IDs are split, not joined
# =============================================================================================

ORDER_ID_FORMS = ["ORD-1004", "ORD 1004", "ORD1004", "o r d dash one zero zero four"]


@pytest.mark.parametrize("form", ORDER_ID_FORMS)
def test_rev_order_id_forms_normalise_to_split_id(form):
    assert _normalize_for_wer(form) == "ord 1004"
    assert _normalize_for_wer(f"My order ID is {form}.") == "my order id is ord 1004"


@pytest.mark.parametrize("reference, hypothesis", list(itertools.permutations(ORDER_ID_FORMS, 2)))
def test_rev_order_id_forms_have_zero_wer_between_any_two(reference, hypothesis):
    assert _corpus_wer([(f"My order ID is {reference}.", f"My order ID is {hypothesis}.")]) == 0.0


def test_rev_customer_id_forms_have_zero_wer():
    assert _corpus_wer([("CUST-2001", "C UST dash 2001")]) == 0.0
    assert _corpus_wer([("It is CUST-2001.", "It is CUST-two thousand 1.")]) == 0.0
    assert _corpus_wer([("CUST2001", "c u s t dash two zero zero one")]) == 0.0


def test_rev_misheard_id_still_counts():
    # Digits and letters of an ID are still compared: one wrong digit group or letter run is an error.
    assert _corpus_wer([("My order ID is ORD-1004.", "My order ID is o r d dash one zero four.")]) > 0
    assert _corpus_wer([("My order ID is ORD-1004.", "My order ID is ORD-1005.")]) > 0
    assert _corpus_wer([("Seat A-7", "Seat H-7")]) > 0


# =============================================================================================
# A4-rev: dates in fixed order (year, month, day); ambiguous numeric dates resolved per pair
# =============================================================================================


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("My date of birth is twelfth April nineteen ninety.", "My date of birth is 04/12/1990."),
        ("1990-04-12", "nineteen ninety zero four twelve"),
        ("2026-10-30", "twenty twenty six ten thirty"),
        ("12th April 1990", "April 12, 1990"),
        ("1990-04-12", "twelfth of April nineteen ninety"),
        # 13 > 12: the order is known (that part is the day)
        ("13/04/1990", "thirteenth April nineteen ninety"),
        ("04/13/1990", "April 13 1990"),
        # ambiguous numeric date (both parts <= 12) resolved to the other side's known-order date
        # with the same year and the same two numbers (either side, either written order)
        ("twelfth April nineteen ninety", "12/04/1990"),
        ("12/04/1990", "twelfth April nineteen ninety"),
        ("12 April 1990", "12-04-1990"),
        ("1990-04-12", "12.04.1990"),
        ("4 December 1990", "04/12/1990"),
        ("04/12/1990", "4 December 1990"),
        # two ambiguous dates written the same way
        ("03/05/2026", "03-05-2026"),
    ],
)
def test_rev_date_pairs_have_zero_wer(reference, hypothesis):
    assert _corpus_wer([(reference, hypothesis)]) == 0.0


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        # review R2-H1: day/month sorting hid these
        ("My date of birth is 4 December 1990.", "My date of birth is 12 April 1990."),
        ("December fourth nineteen ninety", "April twelfth nineteen ninety"),
        ("Pay on March fifth.", "Pay on May third."),
        ("due on 03/05/2026", "due on 05/03/2026"),
        # an ambiguous date is not resolved against a different year
        ("12 April 1990", "12/04/1991"),
        # an ambiguous date is not resolved against different numbers
        ("12 April 1990", "11/04/1990"),
    ],
)
def test_rev_misheard_dates_count_as_errors(reference, hypothesis):
    assert _corpus_wer([(reference, hypothesis)]) > 0
    assert _corpus_wer([(hypothesis, reference)]) > 0


def test_rev_ambiguous_date_resolution_is_per_pair():
    # The first pair resolves (0 errors); the second compares two ambiguous dates as written (1
    # substitution). The resolution of one pair never leaks into another: 1 error / 2 words.
    pairs = [("12/04/1990", "12 April 1990"), ("03/05/2026", "05/03/2026")]
    assert _corpus_wer(pairs) == pytest.approx(0.5)
    pairs = [("12 April 1990", "11 April 1990"), ("12/04/1990", "twelfth April nineteen ninety")]
    assert _corpus_wer(pairs) == pytest.approx(0.5)
    # "04/12/1990" is resolved against "4 December 1990" in its own pair, never against the
    # "12 April 1990" of the other pair.
    pairs = [("12 April 1990", "12 April 1990"), ("04/12/1990", "4 December 1990")]
    assert _corpus_wer(pairs) == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("twelfth April nineteen ninety", "19900412"),
        ("April 12, 1990", "19900412"),
        ("12th April 1990", "19900412"),
        ("Apr 12 1990", "19900412"),
        ("1990-04-12", "19900412"),
        ("2026-10-30", "20261030"),
        ("4 December 1990", "19901204"),
        ("December fourth nineteen ninety", "19901204"),
        ("twelfth april", "0412"),
        ("twelfth of April", "0412"),
        ("March fifth", "0305"),
        ("May third", "0503"),
        # one part > 12: that part is the day
        ("13/04/1990", "19900413"),
        ("04/13/1990", "19900413"),
        ("30-10-2026", "20261030"),
        # ambiguous (both parts <= 12), no other side: year + the two parts in written order
        ("04/12/1990", "19900412"),
        ("12/04/1990", "19901204"),
        ("12-04-1990", "19901204"),
        ("12.04.1990", "19901204"),
        ("03/05/2026", "20260305"),
        ("05/03/2026", "20260503"),
    ],
)
def test_rev_date_tokens(text, expected):
    assert _normalize_for_wer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # may / march / mar are a month only in "<day> of <month>", "<month> <day>" (day as digits or
        # an ordinal word), or when a 4-digit year follows
        ("the fifth of May", "the 0505"),
        ("May 5", "0505"),
        ("May fifth", "0505"),
        ("5 May 2026", "20260505"),
        ("the 3rd of March", "the 0303"),
        ("One may argue that.", "1 may argue that"),
        ("The first may be wrong.", "the 1 may be wrong"),
        ("The second may be fine.", "the 2 may be fine"),
        ("They march on.", "they march on"),
    ],
)
def test_rev_ambiguous_month_words(text, expected):
    assert _normalize_for_wer(text) == expected


# =============================================================================================
# A4-rev: ranges and ordinals are not concatenated
# =============================================================================================


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("Delivery takes 3-5 days.", "Delivery takes 35 days."),
        ("Pages 10-12 please.", "Pages 1012 please."),
        ("I want the first one", "I want the 11"),
        ("I want the first one", "I want the eleventh"),
        ("I want the second one", "I want the 21"),
    ],
)
def test_rev_ranges_and_ordinals_are_not_concatenated(reference, hypothesis):
    assert _corpus_wer([(reference, hypothesis)]) > 0
    assert _corpus_wer([(hypothesis, reference)]) > 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3-5 days", "3 5 days"),
        ("Pages 10-12 please.", "pages 10 12 please"),
        ("the first one", "the 1 1"),
        ("the second one", "the 2 1"),
        ("the first 3 digits", "the 1 3 digits"),
        ("twenty first", "21"),
        ("the eleventh", "the 11"),
    ],
)
def test_rev_range_and_ordinal_outputs(text, expected):
    assert _normalize_for_wer(text) == expected


# =============================================================================================
# A4-rev: no regression against the old normalize_text WER (review R2-H2 and R2-M2)
# =============================================================================================

NO_REGRESSION_PAIRS = [
    ("Enter the one-time password.", "Enter the one time password."),
    ("Enable two-factor authentication.", "Enable two factor authentication."),
    ("This is a first-class service.", "This is a first class service."),
    ("Is it a third-party app?", "Is it a third party app?"),
    ("We offer a 24-hour helpline.", "We offer a 24 hour helpline."),
    ("It is a 5-star hotel.", "It is a five star hotel."),
    ("I tested COVID-19 negative.", "I tested COVID 19 negative."),
    ("Take vitamin B-12 daily.", "Take vitamin B 12 daily."),
    ("My order ID is ORD-1004.", "My order ID is ORD 1004."),
    ("मेरा ऑर्डर ORD-1004 है", "मेरा ऑर्डर ORD 1004 है"),
    ("follow-up", "follow up"),
    ("Am I a good fit?", "Am I the good fit?"),
    ("It is a b grade", "It is a bee grade"),
    ("One may argue that.", "One might argue that."),
    ("The first may be wrong.", "The first might be wrong."),
    ("The second may be fine.", "The second might be fine."),
    ("Oh, one more thing.", "Oh, 1 more thing."),
    ("I'll take the first one please", "I'll take the first won please"),
]


@pytest.mark.parametrize("reference, hypothesis", NO_REGRESSION_PAIRS)
def test_rev_no_more_errors_than_normalize_text(reference, hypothesis):
    new, old = _corpus_wer([(reference, hypothesis)]), _old_wer([(reference, hypothesis)])
    assert new <= old + 1e-9, (new, old)
    new, old = _corpus_wer([(hypothesis, reference)]), _old_wer([(hypothesis, reference)])
    assert new <= old + 1e-9, (new, old)


@pytest.mark.parametrize(
    "reference, hypothesis",
    [pair for pair in NO_REGRESSION_PAIRS[:11]] + [("Oh, one more thing.", "Oh, 1 more thing.")],
)
def test_rev_hyphen_and_number_word_variants_have_zero_wer(reference, hypothesis):
    # A written hyphen is a space and number words become digits, so these are the same words.
    assert _corpus_wer([(reference, hypothesis)]) == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Enter the one-time password.", "enter the 1 time password"),
        ("Enable two-factor authentication.", "enable 2 factor authentication"),
        ("This is a first-class service.", "this is a 1 class service"),
        ("Is it a third-party app?", "is it a 3 party app"),
        ("We offer a 24-hour helpline.", "we offer a 24 hour helpline"),
        ("I tested COVID-19 negative.", "i tested covid 19 negative"),
        ("covid19", "covid 19"),
        ("Take vitamin B-12 daily.", "take vitamin b 12 daily"),
        ("Am I a good fit?", "am i a good fit"),
        ("It is a b grade", "it is a b grade"),
        ("Oh, one more thing.", "oh 1 more thing"),
        ("I'll take the first one please", "ill take the 1 1 please"),
    ],
)
def test_rev_plain_sentence_outputs(text, expected):
    assert _normalize_for_wer(text) == expected


def test_rev_hindi_carrier_sentence_keeps_its_words():
    reference, hypothesis = NO_REGRESSION_PAIRS[9]
    assert _normalize_for_wer(reference) == _normalize_for_wer(hypothesis)
    assert "ord 1004" in _normalize_for_wer(reference)
    assert _normalize_for_wer(reference).split()[0] == "मेरा"


# =============================================================================================
# A4-rev: review R2-L1 (very long digit tokens), R2-L2 (zero before a multiplier), R2-L3
# (non-ASCII decimal digits)
# =============================================================================================

LONG = "1" * 5000


@pytest.mark.parametrize(
    "text",
    [LONG, f"{LONG} hundred", f"{LONG} point five", f"two thousand and {LONG}", f"{LONG} lakh", f"ORD-{LONG}"],
    ids=["digits", "hundred", "point", "after_thousand", "lakh", "id"],
)
def test_rev_very_long_digit_token_never_raises(text):
    out = _normalize_for_wer(text)
    assert LONG in out.replace(" ", "")
    assert _corpus_wer([(text, text)]) == 0.0
    assert _corpus_wer([(text, "one hundred")]) > 0


@pytest.mark.parametrize(
    "text, expected",
    [("0 lakh", "0 lakh"), ("zero lakh", "0 lakh"), ("0 thousand", "0 thousand"), ("10 lakh", "1000000")],
)
def test_rev_zero_before_a_multiplier_is_not_multiplied(text, expected):
    assert _normalize_for_wer(text) == expected


def test_rev_zero_lakh_pair():
    assert _corpus_wer([("0 lakh", "zero lakh")]) == 0.0
    assert _corpus_wer([("0 lakh", "1 lakh")]) > 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("१००४", "1004"),
        ("ORD-१००४", "ord 1004"),
        ("१२ April", "0412"),
        ("५ lakh", "500000"),
        ("pin code ४११००१", "pin code 411001"),
    ],
)
def test_rev_non_ascii_decimal_digits_map_to_ascii(text, expected):
    assert _normalize_for_wer(text) == expected


def test_rev_devanagari_digits_pair_with_ascii_digits():
    assert _corpus_wer([("१००४", "1004")]) == 0.0
    assert _corpus_wer([("My order ID is ORD-१००४.", "My order ID is o r d dash one zero zero four.")]) == 0.0


# =============================================================================================
# A4-rev: accepted residuals (METRICS.md notes)
# =============================================================================================


def test_rev_accepted_residuals():
    # Currency symbols are dropped: a misheard currency scores 0.
    assert _corpus_wer([("Charge ₹500.", "Charge $500.")]) == 0.0
    # A comma does not stop a digit run.
    assert _normalize_for_wer("Account 48291, 3 attempts left.") == "account 482913 attempts left"
    # "ten thirty" is read as the year-like group 1030.
    assert _normalize_for_wer("ten thirty") == "1030"


# =============================================================================================
# normalize_for_wer: the other outputs implied by the rules (A4 steps 1 to 3, unchanged by A4-rev;
# step 5 joins as revised)
# =============================================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        # step 1: separators, decimals, ordinal suffixes, percent, currency, hyphens
        ("45,210", "45210"),
        ("2,500", "2500"),
        ("45210.75", "45210.75"),
        ("12th", "12"),
        ("1st", "1"),
        ("2nd", "2"),
        ("3rd", "3"),
        ("50%", "50 percent"),
        ("₹2,500", "2500"),
        ("$45", "45"),
        ("follow-up", "follow up"),
        ("x-ray", "x ray"),
        ("ORD1004", "ord 1004"),
        # step 2: lowercase, apostrophes, punctuation
        ("I'd like it, please!", "id like it please"),
        # step 3: number words
        ("twenty five", "25"),
        ("two thousand and one", "2001"),
        ("four thousand eight hundred and twenty one", "4821"),
        ("twenty five hundred", "2500"),
        ("four hundred two", "402"),
        ("ten lakh", "1000000"),
        ("10 lakh", "1000000"),
        ("two thousand 1", "2001"),
        ("nineteen ninety", "1990"),
        ("twenty twenty six", "2026"),
        ("nineteen oh five", "1905"),
        ("one oh four", "104"),
        ("seventy five point five", "75.5"),
        ("twenty first", "21"),
        ("one zero zero four", "1004"),
        # step 5 (A4-rev): letter runs, digit runs, spoken dash / hyphen, single letter before letters
        ("o r d", "ord"),
        ("c u s t", "cust"),
        ("a b 12", "ab 12"),
        ("am i a good", "am i a good"),
        ("1 0 0 4", "1004"),
        ("10 01", "1001"),
        ("48291 3", "482913"),
        ("1990 0 4 12", "19900412"),
        ("ORD-1004", "ord 1004"),
        ("ord dash 1004", "ord 1004"),
        ("ord hyphen 1004", "ord 1004"),
        ("cust-two thousand 1", "cust 2001"),
        ("c ust dash 2001", "cust 2001"),
        ("c ust 2001", "cust 2001"),
        ("CU ST dash 2001", "cu st 2001"),
        ("press the dash button", "press the dash button"),
        ("", ""),
    ],
)
def test_normalize_for_wer_outputs(text, expected):
    assert _normalize_for_wer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My order ID is ORD-1004.", "my order id is ord 1004"),
        ("My order ID is o r d dash one zero zero four.", "my order id is ord 1004"),
        ("It is CUST-two thousand 1.", "it is cust 2001"),
        ("My date of birth is 04/12/1990.", "my date of birth is 19900412"),
        ("the OTP 48291 3 that I got", "the otp 482913 that i got"),
        ("payment of 2,500 rupees on twenty twenty six ten thirty", "payment of 2500 rupees on 20261030"),
        ("PIN code four one one zero zero one", "pin code 411001"),
    ],
)
def test_normalize_for_wer_sentences(text, expected):
    assert _normalize_for_wer(text) == expected


# =============================================================================================
# normalize_text is unchanged (VOICE-013 and other callers)
# =============================================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My order ID is ORD-1004.", "my order id is ord 1004"),
        ("2,500 rupees on 2026-10-30", "2 500 rupees on 2026 10 30"),
        ("Twelfth April nineteen ninety", "twelfth april nineteen ninety"),
        ("o r d dash one zero zero four", "o r d dash one zero zero four"),
        ("I'd like 12th, please!", "id like 12th please"),
        ("45210.75", "45210 75"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_text_is_unchanged(text, expected):
    # Regression guard (passes before the fix too).
    from impulse_evals.metrics.voice import normalize_text

    assert normalize_text(text) == expected


@pytest.mark.asyncio
async def test_voice013_still_uses_normalize_text():
    # Regression guard (passes before the fix too): VOICE-013 keeps whole-token matching on
    # normalize_text, so the spelled-out ID is still a missing key term.
    result = await _evaluate(
        "VOICE-013",
        _transcript_from_pairs([("My order ID is ORD-1004.", "My order ID is o r d dash one zero zero four.")]),
        _scenario("voice013_probe", key_terms=["ORD-1004"]),
    )
    assert result.skipped is False
    assert result.raw_value == pytest.approx(1.0)
    assert result.label == "poor"


# =============================================================================================
# Run level: the campaign transcripts recomputed through the metric classes
# =============================================================================================


def _m():
    from impulse_evals import models

    return models


def _scenario(name, **fields):
    values = dict(name=name, language="en-IN", domain="general", max_turns=12, persona="campaign caller")
    values.update(fields)
    return _m().Scenario(**values)


def _transcript_from_pairs(pairs):
    """User turns (text, heard_text), each followed by an agent turn; voice records mark agent audio."""
    turns, records, t = [], [], 0
    for said, heard in pairs:
        turns.append(_m().Turn(speaker="user", text=said, timestamp_ms=t, end_ms=t + 1500, heard_text=heard))
        records.append({"turn_index": len(turns), "speaker": "user", "audio_ms": 1500, "round_trip_text": said})
        turns.append(_m().Turn(speaker="agent", text="Okay.", timestamp_ms=t + 2000, end_ms=t + 3000, latency_ms=500.0))
        records.append({"turn_index": len(turns), "speaker": "agent", "audio_ms": 1000, "stt_text": "Okay."})
        t += 4000
    return _m().CallTranscript(
        turns=turns, duration_ms=t, metadata={"impulse_voice": {"turns": records}},
        scenario_name="voice013_probe", language="en-IN", voice_mode=True,
    )


async def _evaluate(metric_id, transcript, scenario):
    from impulse_evals.judge.base import Judge
    from impulse_evals.metrics.base import MetricContext, get_metric

    class NoJudge(Judge):
        async def evaluate(self, transcript, criterion, tier):  # pragma: no cover - must not be called
            raise AssertionError(f"{metric_id} is deterministic and must not call the judge")

    return await get_metric(metric_id).evaluate(transcript, scenario, NoJudge(), MetricContext(voice_mode=True))


def _campaign(project, run_id):
    data = json.loads((DATA / f"{project}_voice_{run_id}.json").read_text(encoding="utf-8"))
    assert data["run_id"] == run_id
    return data


def _campaign_scenarios(project, run_id):
    """``[(scenario_name, CallTranscript, stored metric labels)]`` of a trimmed campaign run."""
    data = _campaign(project, run_id)
    return [
        (item["scenario_name"], _m().CallTranscript.model_validate(item["transcript"]), item["stored_metrics"])
        for item in data["scenarios"]
    ]


async def _recompute(project, run_id, metric_id):
    results = {}
    for name, transcript, _ in _campaign_scenarios(project, run_id):
        results[name] = await _evaluate(metric_id, transcript, _scenario(name))
    return results


@pytest.mark.parametrize("run_id", [BANKBOT_NORMAL, BANKBOT_BUGGY])
def test_bankbot_fixture_shows_the_false_round_trip_warning(run_id):
    # Regression guard on the fixture (passes before the fix too): the stored runs carry the false
    # warning and at least one poor VOICE-011.
    data = _campaign("bankbot", run_id)
    assert data["voice_mode"] is True
    assert data["stored_round_trip_warning"] is True
    assert any(item["stored_metrics"]["VOICE-011"]["label"] == "poor" for item in data["scenarios"])
    assert [item["scenario_name"] for item in data["scenarios"]] == [
        "balance_with_identity", "lost_card", "caller_offers_otp", "guaranteed_loan",
        "personal_data_request", "data_deletion",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id", [BANKBOT_NORMAL, BANKBOT_BUGGY])
async def test_bankbot_voice011_is_never_poor_after_normalisation(run_id):
    from impulse_evals.metrics.voice import ROUND_TRIP_FAIL

    results = await _recompute("bankbot", run_id, "VOICE-011")
    assert len(results) == 6
    for name, result in results.items():
        assert result.skipped is False, (name, result.reason)
        assert result.informational is True
        assert result.label != "poor", (name, result.raw_value, result.reason)
        assert result.raw_value <= ROUND_TRIP_FAIL, name
    # The runner adds the warning when any counted VOICE-011 result is poor (runner.run); none is.
    assert not any(result.label == "poor" and not result.skipped for result in results.values())


@pytest.mark.asyncio
async def test_bankbot_normal_previously_poor_scenarios_now_round_trip_exactly():
    # balance_with_identity ("C UST dash 2001", "04/12/1990", "CUST-two thousand 1") and
    # caller_offers_otp ("nineteen ninety zero four twelve", "48291 3", "2,500", "twenty twenty six
    # ten thirty") were poor (0.312 and 0.211); every pair is an A4 / A4-rev example shape.
    results = await _recompute("bankbot", BANKBOT_NORMAL, "VOICE-011")
    for name in ("balance_with_identity", "caller_offers_otp"):
        assert results[name].raw_value == pytest.approx(0.0), (name, results[name].reason)
        assert results[name].label == "good"


@pytest.mark.asyncio
async def test_supportbot_voice009_good_where_the_user_was_heard_correctly():
    results = await _recompute("supportbot", SUPPORTBOT_VOICE, "VOICE-009")
    assert set(results) == {"order_status", "refund_within_policy", "damaged_item_ticket"}
    for name, result in results.items():
        assert result.skipped is False, (name, result.reason)
        assert result.raw_value == pytest.approx(0.0), (name, result.reason)
        assert result.label == "good", name


def test_supportbot_fixture_had_poor_voice009():
    # Regression guard on the fixture (passes before the fix too): DEFECT-3's false VOICE-009 results.
    for name, _, stored in _campaign_scenarios("supportbot", SUPPORTBOT_VOICE):
        assert stored["VOICE-009"]["label"] == "poor", name


# =============================================================================================
# End to end: runner.run in voice mode, the round trip hears the ID spelled out
# =============================================================================================

BASE_URL = "http://localhost:8000/v1"
ID_LINE = "My customer ID is CUST-2001."


def _agent_audio(seconds=0.8, sample_rate=16000):
    return wav_bytes(pcm_tone(seconds, sample_rate, frequency_hz=330.0))


@pytest.mark.asyncio
async def test_runner_round_trip_of_a_spelled_id_gives_no_warning(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    from impulse_evals.runner import run

    for _ in range(20):
        mock_voice_server.queue_stt("My customer ID is C UST dash 2001.")
    fake_adapter.replies["spelled_id"] = [
        {
            "text": "Thank you.",
            "audio_bytes": _agent_audio(),
            "latency_ms": 300.0,
            "heard_text": "My customer ID is c u s t dash two zero zero one.",
        }
    ]
    project = impulse_project
    project.write_config(
        [project.scenario("spelled_id", max_turns=1, opening_user_message=ID_LINE, metrics=["VOICE-009"])],
        voice_config={
            "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": BASE_URL},
            "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": BASE_URL},
            "save_audio": False,
            "save_recording": False,
            "timeout_s": 30,
        },
    )
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    metrics = {metric.metric_id: metric for metric in scenario_result.metric_results}
    assert metrics["VOICE-011"].raw_value == pytest.approx(0.0), metrics["VOICE-011"].reason
    assert metrics["VOICE-011"].label == "good"
    assert metrics["VOICE-009"].raw_value == pytest.approx(0.0), metrics["VOICE-009"].reason
    assert metrics["VOICE-009"].label == "good"
    assert ROUND_TRIP_WARNING not in result.warnings


# =============================================================================================
# Re-check N-M1: hyphenated digit chains (phone, Aadhaar, card, toll-free numbers) join like spaces;
# only a single hyphen between two groups of at most 3 digits is a range boundary
# =============================================================================================

HYPHENATED_NUMBER_PAIRS = [
    ("Aadhaar 1234 5678 9012", "Aadhaar 1234-5678-9012"),
    ("My number is 98765 43210.", "My number is 98765-43210."),
    ("Call 1800-123-4567", "Call 1800 123 4567"),
    ("Card 4111-1111-1111-1111", "Card 4111 1111 1111 1111"),
]


@pytest.mark.parametrize("reference, hypothesis", HYPHENATED_NUMBER_PAIRS)
def test_recheck_hyphenated_digit_chain_equals_spaced_number(reference, hypothesis):
    # Review re-check N-M1: these scored 1.5, 0.5, 0.75 and 0.8 after A4-rev (0 with normalize_text).
    assert _corpus_wer([(reference, hypothesis)]) == 0.0
    assert _corpus_wer([(hypothesis, reference)]) == 0.0


def test_recheck_hyphenated_number_equals_spoken_digits():
    pair = ("My number is 98765-43210.", "My number is nine eight seven six five four three two one zero.")
    assert _corpus_wer([pair]) == 0.0
    assert _corpus_wer([(pair[1], pair[0])]) == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1234-5678-9012", "123456789012"),
        ("98765-43210", "9876543210"),
        ("1800-123-4567", "18001234567"),
        ("4111-1111-1111-1111", "4111111111111111"),
        ("Card 4111-1111-1111-1111", "card 4111111111111111"),
        # a chain of short groups is not a range either; nor is a pair with a group longer than 3 digits
        ("1-2-3", "123"),
        ("10-1234", "101234"),
    ],
)
def test_recheck_hyphenated_digit_chain_outputs(text, expected):
    assert _normalize_for_wer(text) == expected


@pytest.mark.parametrize(
    "text, expected, reference, hypothesis",
    [
        ("3-5 days", "3 5 days", "Delivery takes 3-5 days.", "Delivery takes 35 days."),
        ("Pages 10-12 please.", "pages 10 12 please", "Pages 10-12 please.", "Pages 1012 please."),
        ("100-200 rupees", "100 200 rupees", "Pay 100-200 rupees.", "Pay 100200 rupees."),
    ],
)
def test_recheck_two_short_groups_stay_a_range(text, expected, reference, hypothesis):
    # Regression guard (passes before the fix too): the N-M1 change must keep these ranges.
    assert _normalize_for_wer(text) == expected
    assert _corpus_wer([(reference, hypothesis)]) > 0
    assert _corpus_wer([(hypothesis, reference)]) > 0


# =============================================================================================
# Re-check N-L1: a digit-written ordinal is an ordinal like its word form
# =============================================================================================


@pytest.mark.parametrize(
    "digits, words, expected",
    [
        ("the 1st one", "the first one", "the 1 1"),
        ("the 2nd 3 items", "the second 3 items", "the 2 3 items"),
        ("the 21st one", "the twenty first one", "the 21 1"),
        ("flat 5 1st floor", "flat 5 first floor", "flat 5 1 floor"),
        ("on the 3rd 4 5", "on the third 4 5", "on the 3 45"),
    ],
)
def test_recheck_digit_written_ordinal_normalises_like_the_word(digits, words, expected):
    assert _normalize_for_wer(words) == expected   # the word form (unchanged)
    assert _normalize_for_wer(digits) == expected


def test_recheck_digit_written_ordinal_pairs():
    # Review re-check N-L1: "the 1st one" vs "the first one" was 1.0; "the 1st one" vs "the 11" was 0.
    assert _corpus_wer([("the 1st one", "the first one")]) == 0.0
    assert _corpus_wer([("the first one", "the 1st one")]) == 0.0
    assert _corpus_wer([("the 1st one", "the 11")]) > 0
    assert _corpus_wer([("the 11", "the 1st one")]) > 0
    assert _corpus_wer([("on the 2nd 3 items", "on the 23 items")]) > 0


# =============================================================================================
# Re-check R2-L5: no raw private-use characters in the normaliser's source
# =============================================================================================


def test_recheck_wer_normalizer_source_has_no_raw_private_use_characters():
    import impulse_evals

    path = Path(impulse_evals.__file__).resolve().parent / "metrics" / "_wer_normalize.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    found = [
        (number, f"U+{ord(char):04X}")
        for number, line in enumerate(lines, start=1)
        for char in line
        if 0xE000 <= ord(char) <= 0xF8FF
    ]
    assert found == [], f"raw private-use characters (write them as escapes): {found}"
