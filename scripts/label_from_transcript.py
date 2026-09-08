"""Fill the Truth column from linguistic register rather than by listening.

**This is a proxy for hand-labelling, not a substitute.** It reads the transcript and
labels a row by how the speech is *worded* — imperatives, first-person organising, rules,
questions put to a class are a teacher whatever the loudness said. Energy and register are
independent signals, so this is a genuine cross-check on an energy-based split.

Three limitations that must travel with any number produced from it:

1. **It cannot hear.** Tone, distance and overlap are invisible here.
2. **It is biased toward the teacher.** The phone is hers, so her speech is what
   transcribes legibly; a student who spoke from across the room may appear as a row with
   garbled or empty text, which this leaves blank.
3. **The transcript has errors of its own**, so a mis-transcribed row can be mislabelled.

Rows with nothing legible are left blank, and blanks are excluded from scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                              # noqa: E402
from src.attribution import Role                                    # noqa: E402
from src.evaluation.labelling import (                              # noqa: E402
    parse_sheet, render_sheet, with_truth,
)

# Judged from the transcript in eval/labels/*.md. Row ids are session-local suffixes.
# Every one of these is worded as instruction, organisation, a rule, or a question put to
# the class — the register of the person running the room.
TEACHER_ROWS = {
    "OD11163_2025-12-23-121239": [
        "0005",  # long explanation, "समझ में नहीं आ..."
        "0007",  # "अगर कोई भी चीज समझ में नहीं आया तो बताना है" - invites questions
        "0009",  # "बताओ" - imperative
        "0011",  # "आगे जाके हम इलेक्ट्रॉनिक्स के साथ काम करेंगे" - course plan
        "0013",  # "जब हम काम करेंगे" - plural we, instruction
        "0015",  # "संभाल के रखना है ... इस्तेमाल करना है" - a rule
        "0017",  # "जब हम काम करेंगे दूसरे के साथ"
        "0021",  # "आपको दिमाग में ..." - addressing the class
        "0023",  # "तुरंत बताना है आपको" - instruction
        "0025",  # "अगर कोई भी गलती" - rule-setting
        "0031",  # "चलो अभी बताओ की पिछली क्लास में" - prompt
        "0033",  # "क्या क्या कम्पोनेंट ... क्या चीज जरूरी" - question to class
        "0035",  # "कौन सी चीजों का जरूरत पड़ता है" - question to class
    ],
    "OD11163_2026-01-28-121933": [
        "0001",  # "याद कर लेते हैं" - leading a recap
        "0003",  # "का नाम क्या है" - question to class
        "0005",  # the flood story, 145 s of narration
        "0007",  # "हमको कुछ प्लान बनाना पड़ेगा" - sets the task
        "0009",  # "ठीक है क्या" - checking understanding
    ],
    "OD11165_2026-01-06-114155": [
        "0001",  # "स्विच कहाँ होता है" - question to class
        "0005",  # "जब मैं बोलूँगी स्विच ऑफ तो आपको ऐसे बैठना" - first person, giving a rule
        "0007",  # "ठीक है क्या"
        "0011",  # "तुम सब ... बोलना सिखा" - addressing the group
        "0013",  # "चल ये क्या है" - prompt
        "0017",  # "एक चीज नहीं बता दूँ" - offering to explain
    ],
    "OD11166_2026-01-20-115538": [
        "0001",  # "चलो आप ये बताओ की एरोप्लेन कैसे जा रहा था"
        "0003",  # "बोल बोलो" - prompting
        "0005",  # "लिफ्ट क्यों लिखा" - question to class
        "0007",  # explaining lift
        "0013",  # calling on children by name
        "0015",  # "सिद्धि को बुलाता हूँ, कृष्णा को बुलाता हूँ" - first person, organising
        "0017",  # "चलो अभी प्लेन बनाने के लिए"
        "0019",  # "पहला ग्रुप से आध्या" - assigning
        "0021",  # "दूसरे ग्रुप से कुनाल, तीसरे ग्रुप से सिद्धि"
        "0025",  # "कृष्णा तुम" - calling a child
        "0027",  # "सुमित सुमित" - calling a child
        "0029",  # "जिन दिन को पेपर मिला है" - managing materials
    ],
}
STUDENT_ROWS: dict[str, list[str]] = {
    # Nothing legible in these windows reads as a student turn. That is itself a finding -
    # see the bias note in this module's docstring.
}


def main() -> int:
    labels_dir = config.EVAL_DIR / "labels"
    total = filled = 0

    for sheet in sorted(labels_dir.glob("*.md")):
        rows = parse_sheet(sheet.read_text(encoding="utf-8"))
        if not rows:
            continue

        teachers = set(TEACHER_ROWS.get(sheet.stem, []))
        students = set(STUDENT_ROWS.get(sheet.stem, []))

        updated = []
        for r in rows:
            suffix = r.row_id.split("#")[-1]
            truth = (Role.TEACHER if suffix in teachers else
                     Role.STUDENT if suffix in students else None)
            updated.append(with_truth(r, truth))
            total += 1
            filled += truth is not None

        sheet.write_text(render_sheet(updated), encoding="utf-8")
        print(f"  {sheet.stem:<34} {sum(1 for r in updated if r.truth):>3}/{len(updated)} labelled")

    print(f"\n{filled}/{total} rows labelled from transcript register "
          f"({total - filled} left blank as illegible)")
    print("These are a proxy. Spot-check by ear before quoting the accuracy anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
