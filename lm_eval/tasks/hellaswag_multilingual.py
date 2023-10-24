"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf

The Hellaswag dataset translated by the Natural Language Processing Group at the University of Oregon
for [Okapi](https://arxiv.org/abs/2307.16039).
Hellaswag is a commonsense inference challenge dataset. Though its questions are
trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). This is
achieved via Adversarial Filtering (AF), a data collection paradigm wherein a
series of discriminators iteratively select an adversarial set of machine-generated
wrong answers. AF proves to be surprisingly robust. The key insight is to scale up
the length and complexity of the dataset examples towards a critical 'Goldilocks'
zone wherein generated text is ridiculous to humans, yet often misclassified by
state-of-the-art models.

Homepage: https://rowanzellers.com/hellaswag/
"""
from .hellaswag import HellaSwag

_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
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


class HellaSwagMultilingualBase(HellaSwag):
    DATASET_PATH = "jon-tow/okapi_hellaswag"
    DATASET_NAME = None


def create_tasks():
    def _create_task(lang: str):
        class HellaSwagMultilingual(HellaSwagMultilingualBase):
            DATASET_NAME = lang

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def has_training_docs(self):
                return False

            def has_test_docs(self):
                return False

        return HellaSwagMultilingual

    return {f"hellaswag_mt_{lang}": _create_task(lang) for lang in LANGS}
