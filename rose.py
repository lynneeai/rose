"""ACU dataset."""

import os
import json
import datasets

_HOMEPAGE = "https://yale-lily.github.io/ROSE/"

_DESCRIPTION = """
RoSE benchmark
"""

_URL = "https://storage.googleapis.com/sfr-rose-data-research/rose_data.tar.gz"

class ACU(datasets.GeneratorBasedBuilder):
    """ACU dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="cnndm_test"),
        datasets.BuilderConfig(name="cnndm_validation"),
        datasets.BuilderConfig(name="cnndm_protocol"),
        datasets.BuilderConfig(name="cnndm_protocol_gpt3"),
        datasets.BuilderConfig(name="xsum"),
        datasets.BuilderConfig(name="samsum"),
        ]

    def _info(self):
        if self.config.name == "cnndm_test" or self.config.name == "cnndm_protocol":
            self.systems = ['bart', 'gold', 'pegasus', 'brio', 'gsum', 'simcls', 'cliff', 'ctrlsum', 'frost', 'glob', 'matchsum', 'brio-ext']
        elif self.config.name == "cnndm_validation":
            self.systems = ['pegasus', 'gsum', 'matchsum', 'bart', 'brio-ext', 'brio', 'simcls', 'cliff']
        elif self.config.name == "cnndm_protocol_gpt3":
            self.systems = ["bart", "brio", "t0", "gpt3", "reference"]
        elif self.config.name == "xsum":
            self.systems = ['brio', 'frost', 'bart', 'cliff', 'bart.beam_patience', 'pegasus', 'brio-ranking', 'cliff-pegasus']
        elif self.config.name == "samsum":
            self.systems = ['BART', 'PEGASUS', 'MV-BART', 'CODS', 'S-BART', 'PLM-BART', 'Ctrl-DiaSumm', 'UniLM']

        if "protocol" in self.config.name:
            protocol = True
        else:
            protocol = False

        sys_dict = {}
        summary_dict = {}
        for system in self.systems:
            if system != "reference":
                summary_dict[system] = datasets.Value("string")
            if protocol:
                if "gpt" in self.config.name:
                    sys_dict[system] = {"prior": datasets.Value("float32"), "ref_based": datasets.Value("float32"), \
                            "ref_free": datasets.Value("float32"),  "acu_labels": \
                            datasets.features.Sequence(datasets.Value("int64")), \
                            "acu": datasets.Value("float32"), "normalized_acu": datasets.Value("float32")}
                else:
                    sys_dict[system] = {"prior": datasets.Value("int64"), "ref_based": datasets.Value("int64"), \
                            "ref_free": datasets.Value("int64"), "acu_labels": \
                            datasets.features.Sequence(datasets.Value("int64")), \
                            "acu": datasets.Value("float32"), \
                            "normalized_acu": datasets.Value("float32")}
            else:
                sys_dict[system] = {"acu_labels": datasets.features.Sequence(datasets.Value("int64")), \
                        "acu": datasets.Value("float32"), "normalized_acu": datasets.Value("float32")}

        if protocol:
            if "gpt3" in self.config.name:
                features = datasets.Features({"source": datasets.Value("string"), "reference": \
                        datasets.Value("string"), "reference_acus": datasets.features.Sequence(datasets.Value("string")), \
                        "count_id": datasets.Value("int64"), "example_id": \
                        datasets.Value("string"), "annotations": sys_dict, "system_outputs": summary_dict})
            else:
                features = datasets.Features({"source": datasets.Value("string"), \
                        "reference": datasets.Value("string"), "count_id": datasets.Value("int64"), \
                        "example_id": datasets.Value("string"), \
                        "annotations": sys_dict, "system_outputs": summary_dict})
        else:
            features = datasets.Features({"source": datasets.Value("string"), \
                    "reference": datasets.Value("string"), "reference_acus": \
                    datasets.features.Sequence(datasets.Value("string")), "count_id": \
                    datasets.Value("int64"), "example_id": datasets.Value("string"), \
                    "annotations": sys_dict, "system_outputs": summary_dict})
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("source", "reference"),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        files = dl_manager.download_and_extract(_URL)
        if self.config.name.startswith("cnndm"):
            dataset = "cnndm"

        if self.config.name == "cnndm_test":
            split = "test"
            fn = "cnndm.test.acus.aggregated.jsonl"
        elif self.config.name == "cnndm_validation":
            split = "validation"
            fn = "cnndm.val.acus.aggregated.jsonl"
        elif self.config.name == "cnndm_protocol":
            split = "test"
            fn = "cnndm.test.protocols.aggregated.jsonl"
        elif self.config.name == "cnndm_protocol_gpt3":
            split = "test"
            fn = "cnndm.test.protocols-gpt3.aggregated.jsonl"
        elif self.config.name == "xsum":
            dataset = "xsum"
            split = "test"
            fn = "xsum.test.acus.aggregated.jsonl"
        elif self.config.name == "samsum":
            dataset = "samsum"
            split = "test"
            fn = "samsum.test.acus.aggregated.jsonl"

        return [
            datasets.SplitGenerator(
                name="data",
                gen_kwargs={"acu_file": os.path.join(files, f"rose_data/{fn}"), "dataset": dataset, "split": split},
            ),
        ]

    def _generate_examples(self, acu_file, dataset, split):
        """Yields examples."""
        if dataset == "cnndm":
            data_hf = datasets.load_dataset("cnn_dailymail", "3.0.0")[split]
            source_key = "article"
            target_key = "highlights"
        elif dataset == "xsum":
            data_hf = datasets.load_dataset("xsum")[split]
            source_key = "document"
            target_key = "summary"
        elif dataset == "samsum":
            data_hf = datasets.load_dataset("samsum")[split]
            source_key = "dialogue"
            target_key = "summary"

        id2dat = {}
        for count, ex in enumerate(data_hf):
            if dataset == "samsum":
                id2dat[count] = ex
            else:
                id2dat[ex['id']] = ex

        with open(acu_file) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if dataset == "samsum":
                    cur_data_hf = id2dat[data['count_id']]
                else:
                    cur_data_hf = id2dat[data['example_id']]
                data['source'] = cur_data_hf[source_key]
                data['reference'] = cur_data_hf[target_key]
                if self.config.name == "cnndm_protocol_gpt3":
                    data["annotations"]["reference"]["ref_based"] = -1
                    data["annotations"]["reference"]["acu"] = -1
                    data["annotations"]["reference"]["normalized_acu"] = -1
                    data["annotations"]["reference"]["acu_labels"] = []
                yield i, data
