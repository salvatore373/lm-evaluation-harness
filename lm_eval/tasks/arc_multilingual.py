"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC Challenge dataset translated by the Natural Language Processing Group at the University of Oregon
for [Okapi](https://arxiv.org/abs/2307.16039).
The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage:
- https://allenai.org/data/arc
- http://nlp.uoregon.edu/download/okapi-eval/datasets/
"""
from .arc import ARCChallenge


_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
@article{dac2023okapi,
  title={Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author={Dac Lai, Viet and Van Nguyen, Chien and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A and Nguyen, Thien Huu},
  journal={arXiv e-prints},
  pages={arXiv--2307},
  year={2023}
}
"""


LANGS = [
    "ar",
    "bn",
    "ca",
    "da",
    "de",
    "es",
    "eu",
    "fr",
    "gu",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "it",
    "kn",
    "ml",
    "mr",
    "ne",
    "nl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sr",
    "sv",
    "ta",
    "te",
    "uk",
    "vi",
    "zh",
]


class ARCChallengeMultilingualBase(ARCChallenge):
    """TODO: The ARC task relies on `Question:` and `Answer:` formatting which
    will not match multilingual settings. We should translate these for each
    language; note this is not done in Okapi evals:
    https://github.com/nlp-uoregon/mlmm-evaluation/blob/main/lm_eval/tasks/multilingual_arc.py
    """

    DATASET_PATH = "jon-tow/okapi_arc_challenge"
    DATASET_NAME = None


def create_tasks():
    def _create_task(lang: str):
        class ARCChallengeMultilingual(ARCChallengeMultilingualBase):
            DATASET_NAME = lang

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        return ARCChallengeMultilingual

    return {f"arc_challenge_mt_{lang}": _create_task(lang) for lang in LANGS}
