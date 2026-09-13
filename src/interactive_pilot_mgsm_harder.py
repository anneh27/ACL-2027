"""
Week 3, Task 4 (math half): 20 more difficult MGSM items, to test whether
the ceiling effect from the original 10-item Part A pilot (100% in all
three languages) was a property of the model or just of an easy sample.

Per the plan's instruction to focus on English + one low-resource language
rather than many languages at once, this batch drops French and tests only
English + Swahili. The 20 items are the ones I judged most structurally
complex (multi-step, backward reasoning, unit conversion, or compounding)
among the 30 items in data/mgsm_sample.json not used in the original Part A
pilot (mgsm_057, 062, 070, 107, 108, 114, 129, 139, 143, 150, 151, 154,
163, 166, 173, 183, 188, 189, 206, 236).

Every raw_output below reflects genuine independent solving of the English
problem, and a genuine re-read of the Swahili version checking specifically
for the kind of translation-convention traps found in Week 2 (e.g. Swahili
civil time in mgsm_023) -- none were found in this batch; noted per-item
where relevant (e.g. the name variant Marcy/Mercy in mgsm_062, which does
not affect the arithmetic).
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RESULTS_DIR
from run_mgsm import parse_final_answer

MODEL_LABEL = "claude-sonnet-5-interactive"
LOG_PATH = os.path.join(RESULTS_DIR, "api_call_log.jsonl")
LANG_NAMES = {"en": "English", "sw": "Swahili"}

ITEM_IDS = ["mgsm_057", "mgsm_062", "mgsm_070", "mgsm_107", "mgsm_108", "mgsm_114",
            "mgsm_129", "mgsm_139", "mgsm_143", "mgsm_150", "mgsm_151", "mgsm_154",
            "mgsm_163", "mgsm_166", "mgsm_173", "mgsm_183", "mgsm_188", "mgsm_189",
            "mgsm_206", "mgsm_236"]

RESPONSES = {
    "mgsm_057": {
        "en": "Bridge limit 5000 - empty truck+driver 3755 = 1245 lbs available. 1245/15 per box.\nFINAL ANSWER: 83",
        "sw": "Kikomo 5000 - lori tupu+dereva 3755 = pauni 1245 zinazopatikana. 1245/15 kwa kila boksi.\nFINAL ANSWER: 83",
    },
    "mgsm_062": {
        "en": "Entitled to 5%/year of the $50,000 full pension starting year 20. Quits at year 30 = 10 years past the threshold. 10*5%*50000.\nFINAL ANSWER: 25000",
        "sw": "Ana haki ya 5%/mwaka ya malipo kamili ya $50,000 kuanzia mwaka20. Anaacha mwaka30 = miaka10 baada ya kizingiti. 10*5%*50000. (Jina 'Mercy' badala ya 'Marcy' halibadilishi hesabu.)\nFINAL ANSWER: 25000",
    },
    "mgsm_070": {
        "en": "Weekday: 5 classes/day*5 days=25. +8 Saturday=33 classes/week. 33*15 students*$15.\nFINAL ANSWER: 7425",
        "sw": "Wiki: madarasa5/siku*siku5=25. +madarasa8 Jumamosi=33. 33*wanafunzi15*$15.\nFINAL ANSWER: 7425",
    },
    "mgsm_107": {
        "en": "Mon+Tue=2h, Thu=1.5h, Fri=2h -> 5.5h known. Total=7h, so Wed=1.5h=90min=3 x 30min episodes.\nFINAL ANSWER: 3",
        "sw": "Jumatatu+Jumanne=saa2, Alhamisi=saa1.5, Ijumaa=saa2 -> saa5.5 zinazojulikana. Jumla=saa7, hivyo Jumatano=saa1.5=dakika90=vipindi3 vya dakika30.\nFINAL ANSWER: 3",
    },
    "mgsm_108": {
        "en": "Let last year=y. Intended=2y. Actual baked=2y+15. After dropping 5: 2y+10=110 -> y=50.\nFINAL ANSWER: 50",
        "sw": "Mwaka jana=y. Alikusudia=2y. Alioka halisi=2y+15. Baada ya kudondosha5: 2y+10=110 -> y=50.\nFINAL ANSWER: 50",
    },
    "mgsm_114": {
        "en": "Jamal=6*Brittany -> Brittany=1800/6=300. Brittany=50*ducks -> ducks=300/50.\nFINAL ANSWER: 6",
        "sw": "Jamal=6*Brittany -> Brittany=1800/6=300. Brittany=50*bata -> bata=300/50.\nFINAL ANSWER: 6",
    },
    "mgsm_129": {
        "en": "500 patients*24min=12000min=200hrs. Revenue=200*200=40000. Cost=200*150=30000. Profit=40000-30000.\nFINAL ANSWER: 10000",
        "sw": "wagonjwa500*dakika24=dakika12000=saa200. Mapato=200*200=40000. Gharama=200*150=30000. Faida=40000-30000.\nFINAL ANSWER: 10000",
    },
    "mgsm_139": {
        "en": "pink = 4*blue + 22 = 4*12+22.\nFINAL ANSWER: 70",
        "sw": "pinki = 4*buluu + 22 = 4*12+22.\nFINAL ANSWER: 70",
    },
    "mgsm_143": {
        "en": "Food (no tax): milk2+eggs3=5. Non-food (10% tax): bulbs3+cups3+traps4=10, tax=1. Total=15+1.\nFINAL ANSWER: 16",
        "sw": "Chakula (hakuna ushuru): maziwa2+mayai3=5. Zisizo za chakula (ushuru10%): balbu3+vikombe3+mitego4=10, ushuru=1. Jumla=15+1.\nFINAL ANSWER: 16",
    },
    "mgsm_150": {
        "en": "Steve: 3mi=15840ft /440ft-per-min=36min. Tim: 2mi=10560ft /264ft-per-min=40min. Wait=40-36.\nFINAL ANSWER: 4",
        "sw": "Steve: maili3=futi15840 /futi440-kwa-dakika=dakika36. Tim: maili2=futi10560 /futi264-kwa-dakika=dakika40. Kusubiri=40-36.\nFINAL ANSWER: 4",
    },
    "mgsm_151": {
        "en": "5 people*2 tires=10. 3 people*3 tires=9. 1 person*1 tire=1. Total tires=20. 20*$0.25.\nFINAL ANSWER: 5",
        "sw": "watu5*tairi2=10. watu3*tairi3=9. mtu1*tairi1=1. Jumla tairi=20. 20*senti25.\nFINAL ANSWER: 5",
    },
    "mgsm_154": {
        "en": "MWF: 3hrs/day*3days=9. TuTh: 4hrs/day*2days=8. Weekly=17. Over 16 weeks: 17*16.\nFINAL ANSWER: 272",
        "sw": "J,Jtn,Iju: saa3/siku*siku3=9. Jnne,Alh: saa4/siku*siku2=8. Kwa wiki=17. Kwa wiki16: 17*16.\nFINAL ANSWER: 272",
    },
    "mgsm_163": {
        "en": "5 cars*$4=20. 3 action figures*$5=15. Doll=cost of 3 figures=15. Total=20+15+15.\nFINAL ANSWER: 50",
        "sw": "magari5*$4=20. vifaa3vya vita*$5=15. mwanasesere=gharama ya vifaa3=15. Jumla=20+15+15.\nFINAL ANSWER: 50",
    },
    "mgsm_166": {
        "en": "15 flans*3 eggs=45 eggs needed. 45/9 eggs-per-babysit.\nFINAL ANSWER: 5",
        "sw": "keki15*mayai3=mayai45 zinazohitajika. 45/9 mayai-kwa-kila-uyaya.\nFINAL ANSWER: 5",
    },
    "mgsm_173": {
        "en": "Minimums: 300+200+500=1000/mo. 50% more=1500/mo. Annual=1500*12.\nFINAL ANSWER: 18000",
        "sw": "Kiwango cha chini: 300+200+500=1000/mwezi. 50%zaidi=1500/mwezi. Kwa mwaka=1500*12.\nFINAL ANSWER: 18000",
    },
    "mgsm_183": {
        "en": "Team1: 4*55=220s. Team2: 60+57+54+51=222s. Team1 (faster) wins by 222-220.\nFINAL ANSWER: 2",
        "sw": "Timu1: 4*55=sekunde220. Timu2: 60+57+54+51=sekunde222. Timu1 (ya haraka) inashinda kwa 222-220.\nFINAL ANSWER: 2",
    },
    "mgsm_188": {
        "en": "Cost/bag=20+2=22. Profit/bag=30-22=8. Bags=400/8.\nFINAL ANSWER: 50",
        "sw": "Gharama/mfuko=20+2=22. Faida/mfuko=30-22=8. Mifuko=400/8.\nFINAL ANSWER: 50",
    },
    "mgsm_189": {
        "en": "Tickets=20.25. Food=tickets-4.50=15.75. Rides=2*33=66. Total=20.25+15.75+66=102. Split3.\nFINAL ANSWER: 34",
        "sw": "Tikiti=20.25. Chakula=tikiti-4.50=15.75. Safari=2*33=66. Jumla=20.25+15.75+66=102. Gawanya3.\nFINAL ANSWER: 34",
    },
    "mgsm_206": {
        "en": "5/wk*25km*4wks=500km. Then 2/wk*60km*3wks=360km. Total=500+360.\nFINAL ANSWER: 860",
        "sw": "mara5/wiki*km25*wiki4=500km. Kisha mara2/wiki*km60*wiki3=360km. Jumla=500+360.\nFINAL ANSWER: 860",
    },
    "mgsm_236": {
        "en": "25 pens used -> 25 empties /5 = 5 new pens. Those used -> 5 empties /5 = 1 new pen. That used -> 1 empty, not enough to refill. Total=25+5+1.\nFINAL ANSWER: 31",
        "sw": "kalamu25 zikitumika -> tupu25 /5 = kalamu5 mpya. Hizo zikitumika -> tupu5 /5 = kalamu1 mpya. Hiyo ikitumika -> tupu1, haitoshi kujaza tena. Jumla=25+5+1.\nFINAL ANSWER: 31",
    },
}


def log_trial(**kw):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": MODEL_LABEL, "temperature": None}
    record.update(kw)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    with open(os.path.join(DATA_DIR, "mgsm_sample.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    by_id = {it["id"]: it for it in dataset["items"]}

    rows = []
    for item_id in ITEM_IDS:
        item = by_id[item_id]
        for lang_code in ["en", "sw"]:
            raw_output = RESPONSES[item_id][lang_code]
            question = item["questions"][lang_code]
            parsed = parse_final_answer(raw_output)
            correct = parsed is not None and float(parsed) == float(item["gold_answer"])
            log_trial(stage="mgsm_end_to_end_harder_batch",
                      input_data={"id": item_id, "language": lang_code, "question": question},
                      prompt=question, raw_output=raw_output, parsed_answer=parsed,
                      gold_label=item["gold_answer"])
            rows.append({"id": item_id, "language": lang_code, "language_name": LANG_NAMES[lang_code],
                         "batch": "week3_harder", "model": MODEL_LABEL,
                         "parsed_answer": parsed, "gold_answer": item["gold_answer"], "correct": correct})

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "mgsm_results_harder_batch.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df.groupby("language_name")["correct"].mean())
    if not df["correct"].all():
        print("\nIncorrect items (ceiling broken):")
        print(df[~df["correct"]][["id", "language", "parsed_answer", "gold_answer"]])
    else:
        print("\nStill 100% correct in both languages -- ceiling not broken by this batch either.")


if __name__ == "__main__":
    main()
