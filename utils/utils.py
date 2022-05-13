from re import sub
from tkinter import image_names
import pandas as pd
import yaml
import logging
from os import path, listdir

from .errors import DatasetDescriptionError, DataframeGenerationError

_NAME = "name"
_PATH = "path"
_INCLUDE = "include"
_SUBSETS = "subsets"
_ABNORMAL = "abnormal"
_TUBERCULOSIS = "tuberculosis"
_DATA_TYPE = "data_type"
_SUBSET_PATH = "subset_path"
_DATASET_KEYS = [_NAME, _PATH, _INCLUDE, _SUBSETS]
_SUBSET_KEYS = [_INCLUDE, _TUBERCULOSIS, _ABNORMAL, _DATA_TYPE, _SUBSET_PATH]

_FILE = "file"


def load_from_yaml(path, safe_load=True) -> dict:
    try:
        logging.debug("Loading dataset description from yaml file. ")
        with open(path, "r") as file:
            dataset_description = yaml.safe_load(file) if safe_load else yaml.load(file)
            logging.info("Dataset description loadded from file")
            return dataset_description

    except FileNotFoundError:
        raise FileNotFoundError("File {} not found. Please check the path".format(path))

    except Exception as e:
        logging.exception("An error occurred while loading the yaml file")


def _validate_dataset_description(dataset_description) -> bool:
    logging.debug("Validating dataset description")

    if not isinstance(dataset_description, list):
        raise DatasetDescriptionError("Dataset description must be a list")

    for description in dataset_description:
        if not isinstance(description, dict) or not all(
            key in description.keys() for key in _DATASET_KEYS
        ):
            raise DatasetDescriptionError(
                "Dataset description must be a list of dictionaries with keys: {}".format(
                    _DATASET_KEYS
                )
            )
        else:
            for key in _DATASET_KEYS:
                if description.get(key) is None:
                    raise DatasetDescriptionError(
                        "Dataset description key '{}' must not be 'None'".format(key)
                    )

        try:
            for subset in description[_SUBSETS]:
                if not isinstance(subset, dict) or not all(
                    key in subset.keys() for key in _SUBSET_KEYS
                ):
                    raise DatasetDescriptionError(
                        "Subset description must be a list of dictionaries with keys: {}".format(
                            _SUBSET_KEYS
                        )
                    )
                else:
                    for key in _SUBSET_KEYS:
                        if subset.get(key) is None:
                            raise DatasetDescriptionError(
                                "Dataset description key '{}' must not be 'None'".format(
                                    key
                                )
                            )

        except KeyError:
            raise DatasetDescriptionError(
                "Dataset description must contain a list of subsets"
            )

    return True


def generate_dataframe(dataset_description: dict, validade=True) -> pd.DataFrame:
    if validade:
        logging.debug(f"Validate Dataset: {validade}")
        _validate_dataset_description(dataset_description)

    logging.info("Generating dataframe from dataset description")

    dataframe = pd.DataFrame()

    for dataset in dataset_description:
        if dataset.get(_INCLUDE) is True:
            dataset_name = dataset.get(_NAME)
            dataset_path = dataset.get(_PATH)
            logging.info(
                f"Generating dataframe to dataset: {dataset_name}. From path: {dataset_path}"
            )

            for subset in dataset.get(_SUBSETS):
                try:
                    if subset.get(_INCLUDE) is True:
                        subset_relative_path = subset.get(_SUBSET_PATH)
                        logging.info(
                            f"{dataset_name}: Generating dataframe to subset from path: {subset_relative_path}"
                        )

                        subset_full_path = path.join(dataset_path, subset_relative_path)

                        if not path.isdir(subset_full_path):
                            raise DataframeGenerationError(
                                "Unable to find directory: '{}'".format(
                                    subset_full_path
                                )
                            )

                        image_files = listdir(subset_full_path)

                        data_name = []
                        data_path = []
                        data_abnormal = []
                        data_tuberculosis = []

                        for image_file in image_files:
                            if path.splitext(image_file)[1] == subset.get(_DATA_TYPE):

                                # dataframe_temp = dataframe_temp.append(
                                #     {
                                #         _FILE: image_file,
                                #         _PATH: subset_full_path,
                                #         _ABNORMAL: subset.get(_ABNORMAL),
                                #         _TUBERCULOSIS: subset.get(_TUBERCULOSIS),
                                #     },
                                #     ignore_index=True,
                                # )
                                data_name.append(image_file)
                                data_path.append(
                                    path.join(subset_full_path, image_file)
                                )
                                data_abnormal.append(
                                    1 if subset.get(_ABNORMAL) is True else 0
                                )
                                data_tuberculosis.append(
                                    1 if subset.get(_TUBERCULOSIS) is True else 0
                                )

                        dataframe_temp = pd.DataFrame(
                            {
                                _FILE: data_name,
                                _PATH: data_path,
                                _ABNORMAL: data_abnormal,
                                _TUBERCULOSIS: data_tuberculosis,
                            }
                        )

                        dataframe = pd.concat(
                            [dataframe, dataframe_temp], ignore_index=True, axis=0
                        )

                        del dataframe_temp

                        logging.info(
                            f"Dataframe generated from subset: {subset_relative_path}"
                        )

                except Exception as e:
                    logging.exception(
                        f"An error occurred while generating dataframe from subset: {subset.get(_SUBSET_PATH)}. From dataset: {dataset.get(_NAME)}"
                    )

    return dataframe


def load_dataframe(dataframe_path, force_df_gen=False) -> pd.DataFrame:
    logging.debug(f"Force dataframe generation: {force_df_gen}")

    if force_df_gen or not path.isfile(dataframe_path):
        logging.info("Dataframe not found or Force option on.")
        return generate_dataframe(load_from_yaml(dataframe_path), validade=False)

    logging.info("Loading dataframe from file")
    return pd.read_csv(dataframe_path)
