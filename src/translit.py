"""Devanagari -> Modi / Sharada (Unicode). Used to build labels and rendered text."""

def _build():
    ind = "अआइईउऊऋॠऌॡएऐओऔ"
    con = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"
    mat = "ािीुूृॄॢॣेैोौ"
    modi, sha = {}, {}
    for i, c in enumerate(ind):
        modi[c] = chr(0x11600 + i)
        sha[c] = chr(0x11183 + i)
    for i, c in enumerate(con):
        modi[c] = chr(0x1160E + i)                       # same order as Devanagari
        if c == "ळ":
            sha[c] = chr(0x111AD)
        else:
            sha[c] = chr(0x11191 + i + (1 if i >= 28 else 0))   # LLA sits after LA in Sharada
    for i, c in enumerate(mat):
        modi[c] = chr(0x11630 + i)
        sha[c] = chr(0x111B3 + i)
    modi.update({"ं": "\U0001163D", "ः": "\U0001163E", "्": "\U0001163F", "ँ": "\U0001163D",
                 "।": "\U00011641", "॥": "\U00011642", "ऽ": "", "़": "",
                 "ॐ": "\U0001160C\U00011626\U0001163F"})
    sha.update({"ँ": "\U00011180", "ं": "\U00011181", "ः": "\U00011182", "्": "\U000111C0",
                "ऽ": "\U000111C1", "ॐ": "\U000111C4", "।": "\U000111C5", "॥": "\U000111C6",
                "़": ""})
    for d in range(10):
        modi[chr(0x0966 + d)] = chr(0x11650 + d)
        sha[chr(0x0966 + d)] = chr(0x111D0 + d)
    return modi, sha

_MODI, _SHARADA = _build()


def convert(text, script):
    """script in {'devanagari','modi','sharada'}; returns text in that script."""
    if script == "devanagari":
        return text
    table = _MODI if script == "modi" else _SHARADA
    return "".join(table.get(ch, ch) for ch in text)