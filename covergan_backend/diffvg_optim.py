import tempfile

import pydiffvg


def run_diffvg_optim(svg):
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=True) as f:
        f.write(svg)
        x = pydiffvg.svg_to_scene(f.name)
        return pydiffvg.svg_to_str(*x)
