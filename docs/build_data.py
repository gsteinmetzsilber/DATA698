import csv
import json
import random
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

TEST_URL = "https://698modeldata.s3.us-east-2.amazonaws.com/test_persuasive_model_input.csv"
PREDICTIONS_URL = "https://698modelpredictions.s3.us-east-2.amazonaws.com/final_persuasive_predictions_df.csv"
OUTPUT = Path(__file__).with_name("data.js")

ESSAY_SET = "1"
RATERS = ("rater1", "rater2")
LOWEST_SCORE = 1
HIGHEST_SCORE = 6
POOL_SIZE = 20
SEED = 7

PROMPT = [
    "More and more people use computers, but not everyone agrees that this benefits society. "
    "Those who support advances in technology believe that computers have a positive effect on people. "
    "They teach hand-eye coordination, give people the ability to learn about faraway places and people, "
    "and even allow people to talk online with other people. Others have different ideas. "
    "Some experts are concerned that people are spending too much time on their computers and less time "
    "exercising, enjoying nature, and interacting with family and friends.",
    "Write a letter to your local newspaper in which you state your opinion on the effects computers "
    "have on people. Persuade the readers to agree with you.",
]

RUBRIC = [
    (
        "Undeveloped",
        "An undeveloped response that may take a position but offers no more than very minimal support.",
        [
            "Contains few or vague details.",
            "Is awkward and fragmented.",
            "May be difficult to read and understand.",
            "May show no awareness of audience.",
        ],
    ),
    (
        "Under-developed",
        "An under-developed response that may or may not take a position.",
        [
            "Contains only general reasons with unelaborated and/or list-like details.",
            "Shows little or no evidence of organization.",
            "May be awkward and confused or simplistic.",
            "May show little awareness of audience.",
        ],
    ),
    (
        "Minimally-developed",
        "A minimally-developed response that may take a position, but with inadequate support and details.",
        [
            "Has reasons with minimal elaboration and more general than specific details.",
            "Shows some organization.",
            "May be awkward in parts with few transitions.",
            "Shows some awareness of audience.",
        ],
    ),
    (
        "Somewhat-developed",
        "A somewhat-developed response that takes a position and provides adequate support.",
        [
            "Has adequately elaborated reasons with a mix of general and specific details.",
            "Shows satisfactory organization.",
            "May be somewhat fluent with some transitional language.",
            "Shows adequate awareness of audience.",
        ],
    ),
    (
        "Developed",
        "A developed response that takes a clear position and provides reasonably persuasive support.",
        [
            "Has moderately well elaborated reasons with mostly specific details.",
            "Exhibits generally strong organization.",
            "May be moderately fluent with transitional language throughout.",
            "May show a consistent awareness of audience.",
        ],
    ),
    (
        "Well-developed",
        "A well-developed response that takes a clear and thoughtful position and provides persuasive support.",
        [
            "Has fully elaborated reasons with specific details.",
            "Exhibits strong organization.",
            "Is fluent and uses sophisticated transitional language.",
            "May show a heightened awareness of audience.",
        ],
    ),
]


def fetch(url, directory):
    target = Path(directory) / url.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(url, target)
    return target


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_essays(test_path, predictions_path):
    essays = {}
    for row in read_rows(test_path):
        if row["essay_set"] != ESSAY_SET:
            continue
        essay = essays.setdefault(row["essay_id"], {"id": int(row["essay_id"]), "text": row["essay"]})
        for rater in RATERS:
            if row[f"{rater}_domain1"]:
                essay[rater] = int(float(row[f"{rater}_domain1"]))
    for row in read_rows(predictions_path):
        essay = essays.get(row["essay_id"])
        if essay is None:
            continue
        rater = row["active_feature"].removeprefix("score_type_").removesuffix("_domain1")
        essay[f"model_{rater}"] = int(row["prediction"]) + LOWEST_SCORE
    return sorted(essays.values(), key=lambda essay: essay["id"])


def agreement(pairs):
    pairs = list(pairs)
    return {
        "exact": sum(first == second for first, second in pairs),
        "within_one": sum(abs(first - second) <= 1 for first, second in pairs),
        "of": len(pairs),
    }


def quadratic_weighted_kappa(first, second):
    scores = range(LOWEST_SCORE, HIGHEST_SCORE + 1)
    pairs = Counter(zip(first, second))
    first_counts, second_counts = Counter(first), Counter(second)
    observed = sum((a - b) ** 2 * pairs[a, b] for a in scores for b in scores)
    expected = sum((a - b) ** 2 * first_counts[a] * second_counts[b] for a in scores for b in scores) / len(first)
    return 1 - observed / expected


def reference(essays):
    column = lambda key: [essay[key] for essay in essays]
    rater_scores = [essay[rater] for essay in essays for rater in RATERS]
    most_common_score, hits = Counter(rater_scores).most_common(1)[0]
    return {
        "essays": len(essays),
        "model": agreement((essay[f"model_{rater}"], essay[rater]) for essay in essays for rater in RATERS),
        "raters": agreement(zip(column("rater1"), column("rater2"))),
        "constant": {"score": most_common_score, "exact": hits, "of": len(rater_scores)},
        "kappa": {
            "model_rater1": round(quadratic_weighted_kappa(column("model_rater1"), column("rater1")), 3),
            "model_rater2": round(quadratic_weighted_kappa(column("model_rater2"), column("rater2")), 3),
            "raters": round(quadratic_weighted_kappa(column("rater1"), column("rater2")), 3),
        },
    }


def pool(essays):
    return [
        {
            "id": essay["id"],
            "words": len(essay["text"].split()),
            "rater1": essay["rater1"],
            "rater2": essay["rater2"],
            "model": essay["model_rater1"],
            "text": essay["text"],
        }
        for essay in random.Random(SEED).sample(essays, POOL_SIZE)
    ]


def main():
    if len(sys.argv) == 3:
        essays = load_essays(sys.argv[1], sys.argv[2])
    else:
        with tempfile.TemporaryDirectory() as directory:
            essays = load_essays(fetch(TEST_URL, directory), fetch(PREDICTIONS_URL, directory))

    if any(essay["model_rater1"] != essay["model_rater2"] for essay in essays):
        raise SystemExit("The page shows one model score per essay, but the model's two predictions differ.")

    payload = {
        "prompt": PROMPT,
        "rubric": [
            {"score": LOWEST_SCORE + index, "label": label, "summary": summary, "elements": elements}
            for index, (label, summary, elements) in enumerate(RUBRIC)
        ],
        "reference": reference(essays),
        "essays": pool(essays),
    }
    OUTPUT.write_text(
        "window.GRADING_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['essays'])} of {len(essays)} essays to {OUTPUT}")


if __name__ == "__main__":
    main()
