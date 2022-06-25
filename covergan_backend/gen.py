import logging

import yaml

from service import CoverServiceForGenerator, GeneratorType

config = yaml.safe_load(open("config.yml"))


def do_generate(filename, track_artist, track_name, emotions,
                gen_type: str, use_captioner: bool, num_samples: int, use_filters: bool):
    use_color_predictor = False
    if gen_type == "1":
        gen_type = GeneratorType.IlyaGenerator
        gen_weights = config["service"]["gan_weights_ilya"]
        color_predictor_weights = None
    elif gen_type == "2":
        gen_type = GeneratorType.GeneratorFixedSixPaths
        gen_weights = config["service"]["gan_weights_2"]
        color_predictor_weights = None
    elif gen_type == "3":
        gen_type = GeneratorType.GeneratorFixedThreeFigs32
        gen_weights = config["service"]["gan_weights_3"]
        color_predictor_weights = config["service"]["colorer_weights_gan3"]
    elif gen_type == "4":
        gen_type = GeneratorType.GeneratorFixedSixFigs32
        gen_weights = config["service"]["gan_weights_4"]
        color_predictor_weights = config["service"]["colorer_weights_gan4"]
    elif gen_type == "5":
        gen_type = GeneratorType.GeneratorRandFigure
        gen_weights = config["service"]["gan_weights_5"]
        color_predictor_weights = config["service"]["colorer_weights_gan5"]
    else:
        raise Exception(f"Unknown generator type: `{gen_type}`")
    service = CoverServiceForGenerator(
        gen_type,
        gen_weights,
        use_captioner,
        config["service"]["font_dir"],
        color_predictor_weights,
        config["service"]["captioner_weights"],
        log_level=(logging.getLevelName(config["app"]["log_level"]))
    )
    return service.generate(
        filename, track_artist, track_name, emotions,
        num_samples=num_samples, use_color_predictor=use_color_predictor,
        apply_filters=use_filters, watermark=False
    )
