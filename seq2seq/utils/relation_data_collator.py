import random
import warnings
import numpy as np
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        relations = [feature["relations"] for feature in features] if "relations" in features[0].keys() else None
        input_ids = [feature["input_ids"] for feature in features] if "input_ids" in features[0].keys() else None
        
        
        
        # We have to pad the relation matrixs before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if relations is not None:

            sub_len = [len(r)-len(i) for r,i in zip(relations, input_ids)]
            
            assert  not(any(sub_len)), "the relations is not equal with input_ids"

            max_input_ids = max(len(i) for i in input_ids)
            max_relation_length = max(len(r) for r in relations)

            assert max_input_ids==max_relation_length, "max_input_ids is not equal to max_relation_length"

            max_length = max(max_input_ids, max_relation_length)
            for feature in features:
                relation_pad_length = max_length - len(feature['relations']) 
                feature["relations"] = np.pad(np.array(feature["relations"]),((0,relation_pad_length),(0,relation_pad_length)),'constant',constant_values = (0,0))  #constant_values表示填充值，且(before，after)的填充值等于（0,0）

        # print("type(features:)", type(features))
        # print("type(features[0]:)", type(features[0]))
        # print("len(features[0]:)", len(features[0]['input_ids']))
        # print("(features[0]:)", features[0]['input_ids'])

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # print("type(features:)", type(features))
        # print("type(features[0]:)", type(features[0]))
        # print("After pad len(features[0]:)", len(features[0]['input_ids']))
        # print("After pad (features[0]:)", features[0]["input_ids"])
        # import os;os._exit()

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
