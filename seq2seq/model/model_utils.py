from .t5_relation_model import T5ForConditionalGeneration as T5_Relation_debug # 模仿self.shard的处理过程处理relation attention 
from transformers import T5ForConditionalGeneration as T5_Pretrained
from transformers import AutoTokenizer, AutoConfig


def get_relation_t5_model(config, model_name_or_path):
    if 'checkpoint' in model_name_or_path:
        print("use relation model from checkpoint")
        model = T5_Relation_debug.from_pretrained(model_name_or_path)
    else:
        # 返回修改了带有relation的t5模型
        my_config = config
        model = T5_Relation_debug(config=my_config)

        model_pretrained = T5_Pretrained.from_pretrained(model_name_or_path)
        parameter_dict = model_pretrained.state_dict()
        model_dict = model.state_dict()
        model_dict.update(parameter_dict)
        model.load_state_dict(model_dict)

    return model    


def get_pretrained_t5_model():
    '''
        return a t5-small model provided by huggingface transformers lib.
    '''
    model = T5_Pretrained.from_pretrained("t5-small")

    return model


