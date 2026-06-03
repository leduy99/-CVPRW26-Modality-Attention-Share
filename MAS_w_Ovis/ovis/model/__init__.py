from transformers import AutoConfig, AutoModel
from .vit.modeling_siglip2_navit import Siglip2NavitModel
from .vit.configuration_siglip2_navit import Siglip2NavitConfig
from .modeling_ovis import Ovis, OvisPreTrainedModel

AutoConfig.register('siglip2_navit', Siglip2NavitConfig)
AutoModel.register(Siglip2NavitConfig, Siglip2NavitModel)

