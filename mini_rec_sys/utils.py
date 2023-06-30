import datetime
import os
import torch
import pandas as pd
import re
import lxml.html
import lxml.html.clean
from pdb import set_trace


def get_date():
    date = datetime.datetime.now().date()
    return str(date)


def get_time_now():
    timestamp = datetime.datetime.now().time()
    timestamp = str(timestamp).split(".")[0]
    return timestamp


def on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def print_helper(message):
    print(f"{get_time_now()} {message}")


def get_memory():
    if torch.cuda.is_available():
        print_helper(
            f"The GPU usage is: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()*100:.2f}%"
        )


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_num = torch.cuda.current_device()
        print_helper(f"Using the GPU: {torch.cuda.get_device_name(device_num)}")
    else:
        device = torch.device("cpu")
        print_helper(f"Using the CPU.")
    return device


def batcher(iterable, n=1):
    l = len(iterable)
    for idx in range(0, l, n):
        yield iterable[idx : min(idx + n, l)]


def clean(text):
    """
    Clean text by removing html etc.
    Note that we do not lowercase the text as capitalization is often used
    to distinguish acronyms.
    """
    if len(text.strip()) == 0:
        return ""

    # Remove html
    doc = lxml.html.fromstring(text)
    cleaner = lxml.html.clean.Cleaner(style=True)
    doc = cleaner.clean_html(doc)
    text = doc.text_content()

    # Strip beginning and ending whitespaces
    text = text.strip()

    # Remove empty newlines
    text = " ".join([k.strip() for k in text.split("\n") if k])

    # Special cases:
    # s/he = she or he
    text = re.sub(r"([sS]+\s*\/\s*[hH][eE])", "she or he", text)

    # Convert slashes to OR
    text = " or ".join([k.strip() for k in text.split("/")])

    # Replace (s) with the s, e.g. kitchen(s) with kitchens
    text = re.sub(r"\(s\)", "s", text)
    return text


def convert_none_to_empty_string(string):
    if string is None:
        return ""
    return string
