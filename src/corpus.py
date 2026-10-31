"""Source text for each script. Written in Devanagari, converted by translit.convert."""
import numpy as np
from src.translit import convert

HEADER = "श्री गणेशाय नमः ।"

MARATHI_FRAGS = [
    "ऐका श्रोते सावधान", "कथा पुण्य पावन", "ऐकता होय कल्याण", "सकळ जनांसी",
    "एक नगर विख्यात", "राजा धर्मशील तेथ", "प्रजा सुखे नांदत", "राज्यामाजी",
    "राजा करी न्यायदान", "सत्य वचनी निपुण", "साधुसंतांचे पूजन", "नित्य करी",
    "प्रभातीचे समयासी", "राजा गेला देवळासी", "पूजा करूनि भक्तीसी", "नमन केले",
    "तेथे आला एक ऋषी", "तेजस्वी तपोराशी", "राजा लागला चरणासी", "विनयेकरूनि",
    "ऋषी म्हणे राजन", "सांगतो ऐक वचन", "धर्म ठेवी सर्वदा मन", "तरीच सुख",
    "सत्य बोले सर्वकाळ", "दया ठेवी दीनांवर", "तोचि जाणावा ईश्वर", "भक्त होय",
    "ऐकोनि ऋषीचे वचन", "राजा जाहला प्रसन्न", "दिधले बहुत धन", "ब्राह्मणांसी",
    "पुढे झाले ऐका वृत्त", "राजा बैसला सभेत", "प्रधान आला तेथवरी", "विनंती करी राजासी",
]

SANSKRIT_VERSES = [
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ।",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ।",
    "परित्राणाय साधूनां विनाशाय च दुष्कृताम् । धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ।",
    "वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोऽपराणि । तथा शरीराणि विहाय जीर्णान्यन्यानि संयाति नवानि देही ।",
    "देहिनोऽस्मिन्यथा देहे कौमारं यौवनं जरा । तथा देहान्तरप्राप्तिर्धीरस्तत्र न मुह्यति ।",
    "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज । अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः ।",
]


def _dev_num(n):
    return "".join(chr(0x0966 + int(d)) for d in str(n))


def _marathi_block(rng, n_ovi):
    n = int(rng.integers(1, 150))
    out = []
    for _ in range(n_ovi):
        k = int(rng.integers(3, 5))
        frags = rng.choice(MARATHI_FRAGS, size=k, replace=False)
        out.append(" । ".join(frags) + " ।" + _dev_num(n) + "।")
        n += 1
    return " ".join(out)


def _sanskrit_block(rng, n_verse):
    idx = rng.permutation(len(SANSKRIT_VERSES))[:n_verse]
    n = int(rng.integers(1, 60))
    out = []
    for i in idx:
        out.append(SANSKRIT_VERSES[i] + " ॥" + _dev_num(n) + "॥")
        n += 1
    return " ".join(out)


def get_paragraphs(script, rng, n_blocks=3):
    """Returns (paragraphs, devanagari_source). Text is already in the target script."""
    if script == "sharada":
        src = [_sanskrit_block(rng, 2) for _ in range(n_blocks)]
    else:
        src = [_marathi_block(rng, 3) for _ in range(n_blocks)]
    paras = [convert(HEADER, script)] + [convert(p, script) for p in src]
    return paras, src