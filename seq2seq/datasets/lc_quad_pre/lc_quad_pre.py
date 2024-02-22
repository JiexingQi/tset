import json
import re
import datasets
import os
import sys
# from seq2seq.datasets.lc_quad_pre.preprocess import Preprocess


# TODO(lc_quad): BibTeX citation
_CITATION = """
@inproceedings{dubey2017lc2,
title={LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia},
author={Dubey, Mohnish and Banerjee, Debayan and Abdelkawi, Abdelrahman and Lehmann, Jens},
booktitle={Proceedings of the 18th International Semantic Web Conference (ISWC)},
year={2019},
organization={Springer}
}
"""

# TODO(lc_quad):
_DESCRIPTION = """\
LC-QuAD 2.0 is a Large Question Answering dataset with 30,000 pairs of question and its corresponding SPARQL query. The target knowledge base is Wikidata and DBpedia, specifically the 2018 version. Please see our paper for details about the dataset creation process and framework.
"""
# _URL = "https://github.com/AskNowQA/LC-QuAD2.0/archive/master.zip"


class LcQuad(datasets.GeneratorBasedBuilder):
    """TODO(lc_quad): Short description of my dataset."""

    # TODO(lc_quad): Set up version.
    VERSION = datasets.Version("2.0.0")

    def _info(self):
        # TODO(lc_quad): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    # 'uid': datasets.Value("string"),
                    'input_process': datasets.Value("string"),
                    'target_process': datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="http://lc-quad.sda.tech/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_dir = dl_manager.download_and_extract(_URL)
        dl_dir = './transform/transformers_cache/downloads'
        dl_dir = os.path.join(dl_dir, "LC-QuAD2.0-pre", "dataset")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(dl_dir, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(dl_dir, "test.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

            for id_, row in enumerate(data):
                yield id_, {
                    # 'uid': row['uid'],
                    'input_process': row['input'],
                    'target_process': row['target']
                }
