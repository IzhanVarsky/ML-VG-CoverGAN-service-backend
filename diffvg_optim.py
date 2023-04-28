import tempfile

import pydiffvg


def run_diffvg_optim(svg: str):
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=True) as f:
        f.write(svg.encode("utf-8"))
        x = pydiffvg.svg_to_scene(f.name)
        return pydiffvg.svg_to_str(*x)
