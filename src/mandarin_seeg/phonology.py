from __future__ import annotations

import re
from dataclasses import dataclass


INITIALS = (
    "<none>",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
)

FINALS = (
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "er",
    "ia", "io", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "uei", "uan", "uen", "uang", "ueng", "ong",
    "ve", "van", "vn",
)

INITIAL_TO_INDEX = {value: index for index, value in enumerate(INITIALS)}
FINAL_TO_INDEX = {value: index for index, value in enumerate(FINALS)}

_CONSONANT_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s",
)

_Y_FINALS = {
    "": "i", "i": "i", "in": "in", "ing": "ing",
    "a": "ia", "ao": "iao", "e": "ie", "o": "io", "ou": "iou",
    "an": "ian", "ang": "iang", "ong": "iong",
    "u": "v", "ue": "ve", "uan": "van", "un": "vn",
}

_W_FINALS = {
    "": "u", "u": "u", "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
    "an": "uan", "en": "uen", "ang": "uang", "eng": "ueng",
}


@dataclass(frozen=True)
class PhonologyLabel:
    pinyin: str
    syllable: str
    initial: str
    final: str
    tone: int

    @property
    def initial_index(self) -> int:
        return INITIAL_TO_INDEX[self.initial]

    @property
    def final_index(self) -> int:
        return FINAL_TO_INDEX[self.final]


def parse_pinyin(value: str) -> PhonologyLabel:
    text = value.strip().lower().replace("u:", "v").replace("ü", "v")
    match = re.fullmatch(r"([a-zv]+)([1-5])", text)
    if not match:
        raise ValueError(f"unsupported numbered pinyin: {value!r}")
    syllable, tone_text = match.groups()
    tone = int(tone_text) - 1

    if syllable.startswith("y"):
        initial = "<none>"
        remainder = syllable[1:]
        if remainder not in _Y_FINALS:
            raise ValueError(f"unsupported y-initial orthography: {value!r}")
        final = _Y_FINALS[remainder]
    elif syllable.startswith("w"):
        initial = "<none>"
        remainder = syllable[1:]
        if remainder not in _W_FINALS:
            raise ValueError(f"unsupported w-initial orthography: {value!r}")
        final = _W_FINALS[remainder]
    else:
        initial = "<none>"
        final = syllable
        for candidate in _CONSONANT_INITIALS:
            if syllable.startswith(candidate):
                initial = candidate
                final = syllable[len(candidate):]
                break
        if initial in {"j", "q", "x"}:
            final = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}.get(final, final)
        final = {"iu": "iou", "ui": "uei", "un": "uen"}.get(final, final)

    if initial not in INITIAL_TO_INDEX:
        raise ValueError(f"unknown initial {initial!r} from {value!r}")
    if final not in FINAL_TO_INDEX:
        raise ValueError(f"unknown final {final!r} from {value!r}")
    return PhonologyLabel(text, syllable, initial, final, tone)

