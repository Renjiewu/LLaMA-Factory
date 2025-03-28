import json
from enum import Enum
from pathlib import Path

import pandas as pd

from data.dataset_op import DiaType, ImgType
from data.fix_label import fix_label


def convert2submit(test_file: Path, prediction_file: Path, save_path: Path):
    pred_label_list = []

    for line in open(prediction_file, "r"):
        prediction_data = json.loads(line)

        pred_label = prediction_data["predict"]
        pred_label_list.append(pred_label)

    test_data = json.load(open(test_file, "r"))
    save_data = []
    for i, example in enumerate(test_data):
        example["predict"] = pred_label_list[i]
        save_data.append(example)

    df = pd.DataFrame(save_data)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")

def convert2submit_2_list(test_file: Path, prediction_file: Path):
    pred_label_list = []

    for line in open(prediction_file, "r"):
        if line == "\n":
            continue
        prediction_data = json.loads(line)

        pred_label = prediction_data["predict"]
        pred_label_list.append(pred_label)

    test_data = json.load(open(test_file, "r"))
    save_data = []
    for i, example in enumerate(test_data):
        if 'label' in pred_label_list[i]:
            try:
                predict = json.loads(pred_label_list[i])
                example["predict"] = predict.get("label", '')
            except:
                example["predict"] = ""
        else:
            example["predict"] = pred_label_list[i]
        save_data.append(example)

    return save_data

def convert2submit_split(test_file_img: Path, prediction_file_img: Path, test_file_dia: Path, prediction_file_dia: Path, save_path: Path):

    img_data = convert2submit_2_list(test_file_img, prediction_file_img)
    dia_data = convert2submit_2_list(test_file_dia, prediction_file_dia)

    df = pd.DataFrame(img_data+dia_data)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")

def split_result(prediction_file:Path):
    img_pred = []
    dia_pred = []

    for line in open(prediction_file, "r"):
        if line == "\n":
            continue
        prediction_data = json.loads(line)

        pred_label = prediction_data["predict"]
        if "电商领域识图专家" in prediction_data['prompt']:
            img_pred.append(pred_label)
        else:
            dia_pred.append(pred_label)

    return img_pred, dia_pred

def convert_result(test_file:Path, pred_label_list):
    test_data = json.load(open(test_file, "r"))
    save_data = []
    for i, example in enumerate(test_data):
        if 'label' in pred_label_list[i]:
            try:
                predict = json.loads(pred_label_list[i])
                example["predict"] = predict.get("label", '')
            except:
                example["predict"] = ""
        else:
            example["predict"] = pred_label_list[i]
        save_data.append(example)
    return save_data


def convert2submit_split_2(test_file_img: Path, test_file_dia: Path, prediction_file: Path, save_path: Path):
    img_pred, dia_pred = split_result(prediction_file)

    img_data = convert_result(test_file_img, img_pred)
    dia_data = convert_result(test_file_dia, dia_pred)

    # print("*"*20 + "img" + "*"*20 )
    # x = 0   
    for i in img_data:
        i['predict'] = fix_label_op(i['predict'], ImgType)
    # print(x)
    # print("*"*20 + "dia" + "*"*20 )
    # y = 0
    for i in dia_data:
        i['predict'] = fix_label_op(i['predict'], DiaType)
    # print(y)

    # print(f"img_err: {x/len(img_data)}, dia_err: {y/len(dia_data)}, total_err: {(x+y)/(len(img_data)+len(dia_data))}")

    df = pd.DataFrame(img_data+dia_data)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")

def fix_label_op(pred_label: str, label_type: Enum) -> str:
    """
    如果模型预测的意图标签不在意图标签列表中, 计算预测的label和意图标签列表中每个label的相似度, 返回相似度最高的label
    """
    if pred_label not in label_type.__members__.values():
        fixed_label, score = fix_label(pred_label, label_type)
        print(f"pred_label: {pred_label}, fixed_label: {fixed_label}, score: {score}")
        return fixed_label
    return pred_label

def convert2submit_3for1(test_file_img, test_file_dia, prediction_file1, prediction_file2, prediction_file3, save_path):
    img_pred1, dia_pred1 = split_result(prediction_file1)

    img_data1 = convert_result(test_file_img, img_pred1)
    dia_data1 = convert_result(test_file_dia, dia_pred1)

    img_pred2, dia_pred2 = split_result(prediction_file2)

    img_data2 = convert_result(test_file_img, img_pred2)
    dia_data2 = convert_result(test_file_dia, dia_pred2)

    img_pred3, dia_pred3 = split_result(prediction_file3)

    img_data3 = convert_result(test_file_img, img_pred3)
    dia_data3 = convert_result(test_file_dia, dia_pred3)


    for i, j, k in zip(img_data1, img_data2, img_data3):
        fin_label, ex_flag = get_fin_label(i['predict'], j['predict'], k['predict'], ImgType)
        i['predict'] = fin_label
        if not ex_flag:
            print(f"img: {i['image']}")
        # print(111)

    for i, j, k in zip(dia_data1, dia_data2, dia_data3):
        fin_label, ex_flag = get_fin_label(i['predict'], j['predict'], k['predict'], DiaType)
        i['predict'] = fin_label
        if not ex_flag:
            print(f"dia: {i['image']}")
        # print(222)

    df = pd.DataFrame(img_data1+dia_data1)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")

def get_fin_label(pred_label1, pred_label2, pred_label3, label_type):
    # if pred_label1 in label_type.__members__.values():



    if pred_label1 == pred_label2 == pred_label3:
        return fix_label_op(pred_label1, label_type), True
    else:
        print(f"pred_label1: {pred_label1}, pred_label2: {pred_label2}, pred_label3: {pred_label3}")
        if pred_label1 == pred_label2:
            return fix_label_op(pred_label1, label_type), False
        elif pred_label1 == pred_label3:
            return fix_label_op(pred_label1, label_type), False
        elif pred_label2 == pred_label3:
            return fix_label_op(pred_label2, label_type), False
        else:
            return fix_label_op(pred_label2, label_type), False



if __name__ == "__main__":
    # test_file = "data/mire_eval/test1.json"
    # prediction_file = "output/qwen2_vl-7b/lora/sft-infer-v1/generated_predictions.jsonl"
    # save_path = "submit-noise-merge-train.csv"
    # convert2submit(test_file, prediction_file, save_path)

    # img_test_file = "data/mire_enhance/split_eval_image_scene_classification.json"
    # img_prediction_file = "output/qwen2_vl-7b/lora/sft-infer-split-img-r128-noise-epoch6/generated_predictions.jsonl"

    # dia_test_file = "data/mire_enhance/split_eval_dialogue_intent_classification.json"
    # dia_prediction_file = "output/qwen2_vl-7b/lora/sft-infer-split-dia-noise/generated_predictions.jsonl"

    # save_path = "submit-dora-r64.csv"
    # img_test_file = "data/mire_enhance/split_eval_image_scene_classification.json"
    # img_prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r64_img_ep3/generated_predictions.jsonl"

    # dia_test_file = "data/mire_enhance/split_eval_dialogue_intent_classification.json"
    # dia_prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r64_dia_ep3/generated_predictions.jsonl"


    # convert2submit_split(img_test_file, img_prediction_file, dia_test_file, dia_prediction_file, save_path)


    save_path = "submit-dora-r32-explainv3_322_2.csv"
    img_test_file = "data/mire_enhance/split_eval_image_scene_classification.json"
    dia_test_file = "data/mire_enhance/split_eval_dialogue_intent_classification.json"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/merge-lora/sft-infer-lora_r256_ep3/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r256_ep3/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explain/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r256_ep3_explain-2/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/merge-lora/sft-infer-lora_r256_ep3_explain/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r256_ep3_explainv2/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2_banlence/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv22/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r256_ep3_explainv23/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/full/sft-infer-full-ep3/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2_with_full/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep1_explainv2_with_full/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-300step/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-800step/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-v2-322step/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-399/generated_predictions.jsonl"
    # prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-84-2/generated_predictions.jsonl"
    prediction_file = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv3-v2-322-2/generated_predictions.jsonl"


    convert2submit_split_2(img_test_file, dia_test_file, prediction_file, save_path)


    # prediction_file1 = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2-1/generated_predictions.jsonl"
    # prediction_file2 = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2-mask/generated_predictions.jsonl"
    # prediction_file3 = "output/qwen2_vl-7b/dora/sft-infer-dora_r32_ep3_explainv2-noise/generated_predictions.jsonl"


    # convert2submit_3for1(img_test_file, dia_test_file, prediction_file1, prediction_file2, prediction_file3, save_path)



# end main
