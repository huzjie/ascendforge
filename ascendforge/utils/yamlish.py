"""Minimal zero-dependency YAML subset parser.

The managed runtime may not ship PyYAML; this fallback parses the small YAML
subset we actually use (nested maps, lists, scalars, comments, inline lists),
so the CLI works out of the box with no pip installs.
"""
import re


def _strip_comment(line):
    # strip comments only when '#' is preceded by whitespace or start of line
    return re.sub(r"(\s|^)#.*$", "", line).rstrip()


def loads(text):
    lines = text.splitlines()
    root = {}
    # stack items: [indent, container, parent, key]
    stack = [[-1, root, None, None]]
    for raw in lines:
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        # list item
        if content.startswith("- "):
            value = _scalar(content[2:].strip())
            # pop stack until a container at a smaller indent
            while stack and stack[-1][0] >= indent:
                stack.pop()
            cont = stack[-1][1]
            if isinstance(cont, dict):
                # dict of scalars -> convert to list
                for k in list(cont.keys()):
                    cont[k] = cont[k]  # no-op
            if not isinstance(cont, list):
                # replace placeholder dict with list
                parent = stack[-1][3]
                key = stack[-1][2]
                lst = []
                if parent is not None and key is not None:
                    parent[key] = lst
                else:
                    # root became list is unusual; keep dict
                    pass
                stack[-1][1] = lst
                cont = lst
            cont.append(value)
            continue
        # key: value or key:
        m = re.match(r"^(.*?):(?:\s+(.*))?$", content)
        if not m:
            continue
        key = m.group(1).strip()
        rest = (m.group(2) or "").strip()
        # pop stack until smaller indent
        while stack and stack[-1][0] >= indent:
            stack.pop()
        cont = stack[-1][1]
        if rest == "":
            child = {}
            if isinstance(cont, dict):
                cont[key] = child
                stack.append([indent, child, cont, key])
            elif isinstance(cont, list):
                child = {}
                cont.append(child)
                stack.append([indent, child, cont, key])
        else:
            if isinstance(cont, dict):
                cont[key] = _scalar(rest)
            elif isinstance(cont, list):
                cont.append({key: _scalar(rest)})
                stack.append([indent, cont[-1], cont, key])
    return root


def _scalar(s):
    s = s.strip()
    if s in ("null", "~", ""):
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    # inline list [a, b]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x.strip()) for x in inner.split(",")]
    # quoted string
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # int / float
    try:
        if re.match(r"^-?\d+$", s):
            return int(s)
        if re.match(r"^-?\d*\.\d+$", s):
            return float(s)
    except ValueError:
        pass
    return s


def parse(path):
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())
