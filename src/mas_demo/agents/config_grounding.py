"""Check explicit configuration references, not the causal truth of free prose.

Aliases identify configuration fields only; they do not select causes or trials.
Absence checks deliberately cover direct statements in Vietnamese and English.
Other paraphrases and causal plausibility still need content review.
"""
import re
import unicodedata


ALIASES = {
    "optimizer": ("optimizer", "bộ tối ưu"),
    "learning_rate": ("learning rate", "tốc độ học"),
    "momentum": ("momentum",),
    "weight_decay": ("weight decay",),
    "grad_clip_norm": ("grad clip norm", "gradient clipping", "grad clipping", "clipping"),
    "grad_clip_value": ("grad clip value", "value clipping"),
    "label_smoothing": ("label smoothing", "làm mịn nhãn"),
    "batch_size": ("batch size", "kích thước batch"),
    "gradient_accumulation": ("gradient accumulation",),
    "beta1": ("beta1",), "beta2": ("beta2",), "epsilon": ("epsilon",),
    "scheduler": ("scheduler", "lịch học"),
    "scheduler_step_size": ("scheduler step size",),
    "scheduler_gamma": ("scheduler gamma",),
    "scheduler_t_max": ("scheduler t max",),
    "scheduler_patience": ("scheduler patience",),
    "early_stop_patience": ("early stop patience", "early stopping"),
    "hidden_activation": ("hidden activation",),
    "use_batchnorm": ("use batchnorm",),
    "weight_init": ("weight init", "weight initialization"),
}


def _plain(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def explicit_config_references(text):
    """Return mentioned fields and direct statements of absence or activity."""
    text = _plain(text)
    mentioned, absent, active = set(), set(), set()
    # Match longer aliases first so scheduler_t_max does not imply scheduler.
    aliases = sorted(((alias, field) for field, names in ALIASES.items()
                      for alias in {_plain(field), *map(_plain, names)}),
                     key=lambda pair: len(pair[0]), reverse=True)
    masked = text
    for alias, field in aliases:
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if not re.search(pattern, masked):
            continue
        mentioned.add(field)
        before = r"\b(?:khong co|khong dung|khong su dung|thieu|no|without|lacks|missing)\s+"
        after = r"\s+(?:(?:dang|bi)\s+)?(?:khong hoat dong|khong duoc su dung|khong duoc ap dung|bi tat|da tat|is disabled|is off|is inactive|is not used|disabled|absent)\b"
        # A following modifier belongs to the noun/predicate, not to absence:
        # "no momentum instability" and "không hoạt động như mong muốn".
        boundary = r"(?=\s*(?:$|[.,;:!?])|\s+(?:co the|dan den|gay|khien|lam|nen|vi|khi|trong|do|may|can|could|causes?|because|during|in|and|but|is used|is applied)\b)"
        for match in re.finditer("(?:" + before + pattern + "|" + pattern + after + ")" + boundary, masked):
            lead = masked[:match.start()].rstrip()
            if not re.search(r"\b(?:khong|chua|not|never|khong phai|khong he)$", lead):
                absent.add(field)
        enabled = r"\s+(?:(?:van|dang|da)\s+)?(?:hoat dong|duoc su dung|duoc ap dung|duoc bat|is active|is enabled|is used|is applied)\b"
        for match in re.finditer(pattern + enabled, masked):
            if not re.search(r"\b(?:khong|chua|not|never|khong phai|khong he)$", masked[:match.start()].rstrip()):
                active.add(field)
        masked = re.sub(pattern, lambda m: " " * len(m[0]), masked)
    return mentioned, absent, active
