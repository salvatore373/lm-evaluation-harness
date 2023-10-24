"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The MMLU dataset translated by the Natural Language Processing Group at the University of Oregon
for [Okapi](https://arxiv.org/abs/2307.16039).
The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from .hendrycks_test import GeneralHendrycksTest


_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
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


class GeneralHendrycksTestMultilingualBase(GeneralHendrycksTest):
    """TODO: The HendrycksTest task relies on `Question:`, `Choices:`, and
    `Answer:` formatting which will not match multilingual settings. We should
    translate these for each language; note this is not done in Okapi evals:
    https://github.com/nlp-uoregon/mlmm-evaluation/blob/main/lm_eval/tasks/multilingual_mmlu.py
    """

    DATASET_PATH = "jon-tow/okapi_mmlu"
    DATASET_NAME = None


def create_tasks():
    def _create_task(lang: str):
        class HendrycksTestMultilingual(GeneralHendrycksTestMultilingualBase):
            DATASET_NAME = lang

            def __init__(self, *args, **kwargs):
                # Force subject to be `lang` but it's really `all`
                kwargs["subject"] = lang
                super().__init__(*args, **kwargs)

            def has_traing_docs(self):
                return False

            def has_test_docs(self):
                return False

        return HendrycksTestMultilingual

    return {f"hendrycksTest_mt_{lang}": _create_task(lang) for lang in LANGS}
