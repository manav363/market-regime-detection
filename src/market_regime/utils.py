import yaml
import logging

from .config.schema import CONFIG_SCHEMA


def setup_logging(name: str):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    return logging.getLogger(name)


logger = setup_logging("utils")


def load_config(path: str):

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    for section, fields in CONFIG_SCHEMA.items():

        if section not in data:
            logger.warning(f"Config missing section: {section}")
            continue

        for field in fields:
            if field not in data[section]:
                logger.warning(f"Missing key: {section}.{field}")

    return data