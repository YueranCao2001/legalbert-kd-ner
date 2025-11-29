import os
import re
from datasets import load_dataset


def extract_entities(example):
    """
    从一个样本中提取实体：
    返回列表 [(start, end, label), ...]
    start/end 为字符位置（半开区间），label 为字符串，例如 'STATUTE'
    """
    entities = []

    for ann in example.get("annotations", []):
        for res in ann.get("result", []):
            value = res.get("value", {})
            start = value.get("start")
            end = value.get("end")
            labels = value.get("labels") or []

            if start is None or end is None or not labels:
                continue

            label = labels[0]
            entities.append((int(start), int(end), str(label)))

    entities.sort(key=lambda x: x[0])
    return entities


def get_text(example):
    """
    统一获取文本（string），兼容以下情况：
    1) example["data"]["text"] 是 string
    2) example["data"]["text"] 是 list of strings（字符级或 token 级）
    3) example["meta"]["data"]["text"]
    4) example["text"]
    """

    # ---- case 1 & 2: example["data"]["text"] ----
    data = example.get("data")
    if isinstance(data, dict):
        txt = data.get("text")

        # 如果是 string
        if isinstance(txt, str):
            return txt

        # 如果是 list -> join 成 string
        if isinstance(txt, list):
            # 确保每个元素都是可拼接的
            parts = []
            for x in txt:
                if isinstance(x, str):
                    parts.append(x)
                else:
                    parts.append(str(x))  # 兜底
            return "".join(parts)

    # ---- case 3: example["meta"]["data"]["text"] ----
    meta = example.get("meta")
    if isinstance(meta, dict):
        d = meta.get("data", {})
        txt = d.get("text")
        if isinstance(txt, str):
            return txt
        if isinstance(txt, list):
            parts = []
            for x in txt:
                parts.append(str(x))
            return "".join(parts)

    # ---- case 4: top-level text ----
    txt = example.get("text")
    if isinstance(txt, str):
        return txt
    if isinstance(txt, list):
        return "".join(str(x) for x in txt)

    return ""


def text_to_conll(text, entities, f):
    """
    用简单空白分词把 text 切成 token，
    根据实体 span 生成 BIO 标注，写入句子。
    """
    text = text or ""
    text = text.strip()
    if not text:
        return 0  # 没有有效文本，返回 0 句

    tokens = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        start, end = m.start(), m.end()
        tokens.append((tok, start, end))

    if not tokens:
        return 0

    for tok, start, end in tokens:
        tag = "O"
        for es, ee, label in entities:
            if end <= es or start >= ee:
                continue
            prefix = "B" if start == es else "I"
            tag = f"{prefix}-{label}"
            break

        f.write(f"{tok} {tag}\n")

    f.write("\n")
    return 1  # 写了一句


def write_split(dataset_split, path, split_name="split"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    num_sentences = 0
    num_examples = 0

    with open(path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            num_examples += 1
            text = get_text(example)
            entities = extract_entities(example)
            num_sentences += text_to_conll(text, entities, f)

    print(f"{split_name}: {num_examples} examples, {num_sentences} sentences written to {path}")


def main():
    print("Downloading InLegalNER from HuggingFace…")
    dataset = load_dataset("opennyaiorg/InLegalNER")
    print("Available splits:", list(dataset.keys()))

    # ---- 选择 train / dev / test ----
    train_split = dataset["train"]

    # dev: 优先用现成的 validation/dev，否则从 train 切 10%
    if "validation" in dataset:
        dev_split = dataset["validation"]
    elif "dev" in dataset:
        dev_split = dataset["dev"]
    else:
        print("No explicit dev/validation split found, creating 10% dev from train...")
        tmp = train_split.train_test_split(test_size=0.1, seed=42)
        train_split = tmp["train"]
        dev_split = tmp["test"]

    # test: 若没有，就用 dev 顶上（极端 fallback）
    if "test" in dataset:
        test_split = dataset["test"]
    else:
        print("No explicit test split found, using dev as test (fallback).")
        test_split = dev_split

    # ---- 写成 CoNLL ----
    print("Generating train split...")
    write_split(train_split, "data/raw/inlegalner/train.conll", split_name="train")

    print("Generating dev split...")
    write_split(dev_split, "data/raw/inlegalner/dev.conll", split_name="dev")

    print("Generating test split...")
    write_split(test_split, "data/raw/inlegalner/test.conll", split_name="test")

    print("Done! Saved to data/raw/inlegalner/")


if __name__ == "__main__":
    main()
