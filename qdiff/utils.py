import logging
import torch
import torch.nn as nn
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer

logger = logging.getLogger(__name__)


def convert_adaround(model):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of layer {}'.format(name))
                continue
            else:
                # logger.info('Change layer {} to adaround'.format(name))
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                   weight_tensor=module.org_weight.data)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of block {}'.format(name))
                continue
            else:
                # logger.info('Change block {} to adaround'.format(name))
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        if sub_module.split != 0:
                            # print(f"split {name}")
                            sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                    weight_tensor=sub_module.org_weight.data[:, :sub_module.split, ...])
                            sub_module.weight_quantizer_0 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_0, round_mode='learned_hard_sigmoid',
                                                                    weight_tensor=sub_module.org_weight.data[:, sub_module.split:, ...])
                        else:
                            sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                    weight_tensor=sub_module.org_weight.data)
        else:
            convert_adaround(module)


def resume_cali_model(qnn, ckpt_path, cali_data, quant_act=False, act_quant_mode='qdiff', cond=False):
    print("Loading quantized model checkpoint")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("Initializing weight quantization parameters")
    qnn.set_quant_state(True, False)
    if not cond:
        cali_xs, cali_ts = cali_data
        _ = qnn(cali_xs.cuda(), cali_ts.cuda())
    else:
        cali_xs, cali_ts, cali_cs = cali_data
        _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_cs.cuda())
    # change weight quantizer from uniform to adaround
    convert_adaround(qnn)
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)

    # remove act_quantizer states for now
    keys = [key for key in ckpt.keys() if "act" in key]
    for key in keys:
        del ckpt[key]
    qnn.load_state_dict(ckpt, strict=(act_quant_mode=='qdiff'))
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zero_data = m.zero_point.data
            delattr(m, "zero_point")
            m.zero_point = zero_data

            delta_data = m.delta.data
            delattr(m, "delta")
            m.delta = delta_data

    if quant_act:       
        print("Initializing act quantization parameters")
        qnn.set_quant_state(True, True)
        if not cond:
            _ = qnn(cali_xs.cuda(), cali_ts.cuda())
        else:
            _ = qnn(cali_xs.cuda(), cali_ts.cuda(), cali_cs.cuda())
        print("Loading quantized model checkpoint again")
        
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                m.zero_point = nn.Parameter(m.zero_point)
                m.delta = nn.Parameter(m.delta)
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    if not torch.is_tensor(m.zero_point):
                        m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                    else:
                        m.zero_point = nn.Parameter(m.zero_point)
                    
        ckpt = torch.load(ckpt_path, map_location='cpu')
        qnn.load_state_dict(ckpt)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                zero_data = m.zero_point.data
                delattr(m, "zero_point")
                m.zero_point = zero_data

                delta_data = m.delta.data
                delattr(m, "delta")
                m.delta = delta_data
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    zero_data = m.zero_point.item()
                    delattr(m, "zero_point")
                    assert(int(zero_data) == zero_data)
                    m.zero_point = int(zero_data)