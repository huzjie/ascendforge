"""Benchmark report rendering (markdown + text)."""


def render_report(results, title="ascendforge benchmark"):
    lines = [f"# {title}", ""]
    for r in results:
        lines.append(f"## {r.get('op', '?')}")
        for k, v in r.items():
            if k == "op":
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)
