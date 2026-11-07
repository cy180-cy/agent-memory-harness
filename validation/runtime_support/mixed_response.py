"""Conservative, deterministic evidence for audit-only lookup preambles.

This is not semantic extraction. Unrecognized clauses, answers, consent,
commitments and mixed write operations fail closed. No task/model identifiers
or private response strings are used by the rules.
"""

import re

VERSION = "mixed-lookup-preamble-v1"


def _plain(value):
    return " ".join(value.replace("’", "'").split()).casefold()


def transition_audit(content, calls, tools, messages):
    """Require every clause to be courtesy, a lookup intention or cited input.

    Tentative names are candidate lookup arguments, never confirmed identity or
    authorization. Their components must also occur in a visible user handle.
    The returned audit is never injected into model messages.
    """
    audit = {"version": VERSION, "status": "blocked", "reason": "mixed_content_not_safely_expressible",
             "original_content": content, "destination": "audit_only_not_model_context",
             "tool_call_ids": [c.get("id") for c in calls], "clauses": []}
    if not isinstance(content, str) or not content.strip() or "?" in content:
        return audit
    available = {t.name: t for t in tools}
    arguments, ambiguous = {}, set()
    for call in calls:
        tool = available.get(call["name"])
        func = getattr(tool, "_func", None)
        if (getattr(func, "__tool_type__", None) != "read"
                or getattr(func, "__mutates_state__", None) is not False):
            audit["reason"] = "mixed_non_read_tool_requires_explicit_text_handling"
            return audit
        for key, value in call["arguments"].items():
            if key in arguments and arguments[key] != value:
                ambiguous.add(key)
            arguments[key] = value
    # Multiple calls may legitimately use the same parameter with different
    # values. Such a parameter cannot ground prose, but calls keep their order.
    arguments = {key: value for key, value in arguments.items() if key not in ambiguous}
    users = []
    for index, message in enumerate(messages or []):
        value = message if isinstance(message, dict) else message.model_dump()
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            users.append((index, _plain(value["content"])))
    user_index, user_text = users[-1] if users else (None, "")
    fields = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            label = key.replace("_", " ")
            fields[label] = value
            if key == "zip":
                fields["zip code"] = value
    # A composite name is still just the unconfirmed arguments of a READ call.
    if all(k in arguments and isinstance(arguments[k], str) for k in ("first_name", "last_name")):
        fields["name"] = arguments["first_name"] + " " + arguments["last_name"]

    def lookup_intent(sentence):
        start = r"(?:let me(?: try to)?|i'll|i will|i can help) (?:find|locate|look up|check) (?:your|the) (?:account|order|information|details)"
        match = re.fullmatch(start + r"(?: using (.+))?", sentence)
        if not match:
            return False
        query = match.group(1)
        if query is None or query == "that information":
            return True
        labels = sorted(fields, key=len, reverse=True)
        label_pattern = "(?:" + "|".join(re.escape(k) for k in labels) + ")"
        if labels and re.fullmatch(r"your " + label_pattern + r"(?: and (?:your )?" + label_pattern + r")*", query):
            return True
        return any(query == f"that information along with your {label} ({_plain(value)})"
                   for label, value in fields.items())

    clauses = [part.strip() for part in re.split(r"[.!]+", _plain(content)) if part.strip()]
    for sentence in clauses:
        evidence = {"text": sentence}
        if sentence in ("thank you", "thanks", "thank you for providing that information", "thank you for your patience"):
            evidence["kind"] = "courtesy"
        elif lookup_intent(sentence):
            evidence["kind"] = "lookup_intention"
        else:
            missing = re.fullmatch(r"since you (?:don't|do not) remember your (email address|email|name|zip code), (.+)", sentence)
            tentative = re.fullmatch(r"based on your (username|user name|handle), i believe your name is (.+)", sentence)
            if missing and lookup_intent(missing[2]) and re.search(
                    r"\bi (?:don't|do not)(?: actually)? remember (?:the|my) " + re.escape(missing[1]) + r"\b", user_text):
                evidence.update(kind="visible_user_restatement_and_lookup", source_message_index=user_index)
            elif tentative and "name" in fields and tentative[2] == _plain(fields["name"]):
                handle = re.search(r"\bmy (?:username|user name|handle) is ([a-z0-9_\-]+)\b", user_text)
                parts = re.findall(r"[a-z]+", tentative[2])
                source_parts = re.findall(r"[a-z]+", handle[1]) if handle else []
                if not parts or parts != source_parts:
                    return audit
                evidence.update(kind="tentative_lookup_arguments_not_confirmed_identity",
                                source_message_index=user_index, argument_keys=["first_name", "last_name"])
            else:
                return audit
        audit["clauses"].append(evidence)
    if clauses:
        audit.update(status="normalized", reason=None)
    return audit
