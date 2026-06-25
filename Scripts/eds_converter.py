import logging
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple, List, AnyStr

# Configure logger for this module
logger = logging.getLogger("eds_converter")


def _convert_bytes_to_str(value: Any) -> Any:
    """
    Recursively convert bytes objects to strings, handling nested data structures.

    This function processes various data types and converts any bytes or numpy.bytes_
    objects to UTF-8 decoded strings. It handles nested structures like lists,
    tuples, NumPy arrays, and dictionaries by recursively applying the conversion
    to their elements or values.

    Parameters
    ----------
    value : Any
        The value to process, which may be bytes, numpy.bytes_, a list/tuple/ndarray
        containing bytes, a dictionary with bytes values, or other types.

    Returns
    -------
    Any
        The processed value with all bytes objects converted to strings where applicable.
        For complex structures, returns a new structure with converted elements.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    elif isinstance(value, np.bytes_):
        return value.decode("utf-8")
    elif isinstance(value, (list, tuple, np.ndarray)):
        # Process each element recursively.
        # Note: We do HAVE to unwrap single-element containers to properly convert them into values.
        if len(value) == 1:
            return _convert_bytes_to_str(value[0])
        return [_convert_bytes_to_str(item) for item in value]
    elif isinstance(value, dict):
        # Recursively process dictionary values.
        return {k: _convert_bytes_to_str(v) for k, v in value.items()}
    else:
        return value


def get_value_from_dict(dictionary: Dict, key_list: List) -> Any:
    """
    Return a value from a nested dictionary given a sequenced list of keys.

    Parameters
    ----------
    dictionary : Dict
        The nested dictionary to search.
    key_list : List
        A list of keys representing the path to the desired value.

    Returns
    -------
    Any
        The value found at the end of the key path, or None if any key in the path
        is missing or the structure is not a dictionary.
    """
    value = dictionary
    for key in key_list:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return None
        else:
            # Path is broken because we encountered a non-dictionary object
            return None
    return value


def str2dict(string: Union[str, None]) -> Dict[str, Any]:
    """
    Convert a JSON-formatted string representation of a dictionary into an actual Python dictionary.

    This function attempts to parse a string that represents a dictionary in JSON format.
    If the string contains nested dictionaries represented as strings, those will also be converted.

    Parameters
    ----------
    string : str or None
        A string representation of a dictionary in JSON format.

    Returns
    -------
    Dict[str, Any]
        A Python dictionary parsed from the input string. Returns an empty dictionary if
        the input is None, empty, or cannot be parsed as a dictionary.
    """
    if not isinstance(string, str) or not string:
        return {}

    # Standardize quotes to double quotes for JSON compatibility.
    # Warning: This is a simple heuristic and may fail if strings contain escaped single quotes.
    processed_string = string.replace("'", '"')

    try:
        dictionary = json.loads(processed_string)
    except json.JSONDecodeError as e:
        logger.error(
            f"Could not convert string to dictionary using json: {e}. String: {string}"
        )
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error converting string to dictionary: {e}. String: {string}"
        )
        return {}

    if isinstance(dictionary, dict):
        # Recursively convert values that look like JSON dictionaries or lists.
        for key, value in dictionary.items():
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed.startswith(("{", "[")):
                    try:
                        dictionary[key] = str2dict(value)
                    except Exception:
                        pass
        return dictionary

    return {}


def get_attr(dataset: h5py.Dataset) -> Dict:
    """
    Retrieve the attributes of an HDF5 dataset as a dictionary with processed values.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 dataset to extract attributes from.

    Returns
    -------
    Dict
        A dictionary of attributes where bytes are converted to strings and
        JSON strings are converted to dictionaries.
    """
    attrs = dict(dataset.attrs) if dataset.attrs else {}
    processed_attrs = {}
    for key, value in attrs.items():
        # Convert bytes/numpy-bytes to str first.
        converted_val = _convert_bytes_to_str(value)
        if isinstance(converted_val, str):
            # Try to parse as dictionary if it looks like JSON.
            trimmed = converted_val.strip()
            if trimmed.startswith(("{", "[")):
                processed_attrs[key] = str2dict(converted_val)
            else:
                processed_attrs[key] = converted_val
        else:
            processed_attrs[key] = converted_val

    return processed_attrs


def load_spectrum_image(path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load multiple HDF5 parts of a spectrum image and concatenate them.

    Parameters
    ----------
    path : Path
        The directory containing the 'Part-*.hdf5' files.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        A tuple containing the concatenated data array and a dictionary of metadata
        from each part.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If no 'Part-*.hdf5' files are found in the directory.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find expected Spectral imaging directory: {path}"
        )

    parts = []
    metadata_dict = {}

    # Use glob to find parts in sorted order to ensure spatial/spectral consistency.
    files = sorted(list(path.glob("Part-*.hdf5")))

    if not files:
        logger.error(f"No files matching 'Part-*.hdf5' found in {path}")
        raise ValueError(f"No 'Part-*.hdf5' files found in {path}")

    logger.info(f"Found {len(files)} spectrum imaging parts in {path}")

    for f in files:
        try:
            with h5py.File(f, "r") as part_file:
                logger.debug(f"Loading file {f.name}")
                dataset = part_file["DefaultDataset"]
                parts.append(np.squeeze(np.array(dataset)))
                metadata_dict[f.stem] = get_attr(dataset)
        except Exception as e:
            logger.error(f"Failed to load part file {f}: {e}")
            raise

    data = np.concatenate(parts, axis=0).T
    logger.info("Successfully concatenated spectrum imaging parts.")
    return data, metadata_dict


def set_metadata_from_JH5metadata(
    signal, metadata: Dict
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Sets microscope parameters on the signal based on a JH5 metadata dictionary.

    Parameters
    ----------
    signal : Any
        The hyperspy/exspy signal object.
    metadata : Dict
        The metadata dictionary extracted from the JH5 file.

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[float]]
        The extracted (ht, live_time, real_time) values.
    """
    ht_keys = [
        "JH5",
        "OptionalData",
        "Information",
        "Tags",
        "HT",
        "AccelerationVoltage",
    ]
    live_time_keys = [
        "JH5",
        "OptionalData",
        "Information",
        "Tags",
        "EDS",
        "DetectorInformation",
        "FirstDetector",
        "LiveTime",
    ]
    real_time_keys = [
        "JH5",
        "OptionalData",
        "Information",
        "Tags",
        "EDS",
        "DetectorInformation",
        "FirstDetector",
        "RealTime",
    ]

    ht = get_value_from_dict(metadata, ht_keys)
    live_time = get_value_from_dict(metadata, live_time_keys)
    real_time = get_value_from_dict(metadata, real_time_keys)

    if ht is not None:
        try:
            # Beam energy is typically passed in kV or eV depending on the signal type.
            # Following original logic: beam_energy = ht * 1e-3
            beam_energy = float(ht) * 1e-3
            signal.set_microscope_parameters(
                beam_energy=beam_energy, live_time=live_time, real_time=real_time
            )
            logger.debug(
                f"Set microscope parameters: beam_energy={beam_energy}, live_time={live_time}, real_time={real_time}"
            )
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Could not set microscope parameters due to type error: {e}"
            )
    else:
        logger.warning(
            "Acceleration voltage (HT) not found in metadata; skipping microscope parameter setup."
        )

    return (ht, live_time, real_time)


def set_calibration_from_JH5metadata(signal, metadata: Dict) -> Optional[List]:
    """
    Sets the axis calibrations of the signal based on a JH5 metadata dictionary.

    Parameters
    ----------
    signal : Any
        The hyperspy/exspy signal object.
    metadata : Dict
        The metadata dictionary extracted from the JH5 file.

    Returns
    -------
    Optional[List]
        The calibration coefficients list if found, else None.
    """
    calibration_keys = [
        "JH5",
        "OptionalData",
        "Information",
        "MeasurementInformation",
        "CalibrationCoefficients",
    ]
    calibration = get_value_from_dict(metadata, calibration_keys)

    if calibration is None:
        logger.warning("Calibration coefficients not found in metadata.")
        return None

    try:
        for ax, cal in enumerate(calibration):
            unit = cal.get("Unit", "unknown")
            # Normalize unit case: first letter lower, rest as is (e.g., "KEV" -> "keV")
            unit = unit[0].lower() + unit[1:] if unit else "unknown"

            scale = cal.get("Scale", 1.0)
            offset = cal.get("Offset", 0.0)

            signal.axes_manager[ax].scale = scale
            signal.axes_manager[ax].units = unit
            signal.axes_manager[ax].offset = offset
            signal.axes_manager[ax].convert_to_units()

            logger.debug(
                f"Axis {ax} calibration set: scale={scale}, offset={offset}, unit={unit}"
            )
    except Exception as e:
        logger.error(f"Error applying calibration to signal: {e}")
        raise

    return calibration


def load_EDSjh5(path: Union[str, Path]):
    """
    Load a jh5 EDS file and its associated spectrum image if present.

    Parameters
    ----------
    path : Union[str, Path]
        The path to a .jh5 EDS file.

    Returns
    -------
    signal : hs.signals.Signal2D
        The resulting EDS signal (either the summed image or the full spectrum image).

    Raises
    ------
    ValueError
        If the file does not have a .jh5 extension.
    """
    import hyperspy.api as hs

    path = Path(path)
    if path.suffix != ".jh5":
        raise ValueError(
            f'Cannot load file "{path.absolute()}". Can only load .jh5 files.'
        )

    logger.info(f"Loading JH5 file: {path.name}")

    try:
        with h5py.File(path, "r") as jh5_file:
            # Dataset "0" typically contains the summed spectrum
            dataset = jh5_file["0"]
            data = np.squeeze(np.array(dataset))
            metadata_dict = get_attr(dataset)
    except Exception as e:
        logger.error(f"Failed to read JH5 file {path}: {e}")
        raise

    signal = hs.signals.Signal2D(data)
    signal.original_metadata.add_dictionary({"JH5": metadata_dict})

    datatype = metadata_dict.get("DataType")
    if datatype == "EdsCube":
        signal.set_signal_type("EDS_TEM")
        logger.debug("Signal type set to EDS_TEM based on DataType='EdsCube'")

    signal.metadata.General.title = path.stem

    # Check for EDS spectrum data location in subdirectory
    try:
        expected_path = path.parent / path.stem / "WL" / "Spectralimaging"
        spectrum_image, spectrum_metadata_dict = load_spectrum_image(expected_path)

        logger.info("Found spectrum image in subdirectory. Loading full data cube...")

        spectrum_signal = hs.signals.Signal2D(spectrum_image).T
        spectrum_signal.set_signal_type("EDS_TEM")

        # The original metadata of the full cube should include the summed signal's info
        spectrum_metadata_dict["Summed"] = signal
        spectrum_signal.original_metadata.add_dictionary(
            {"JH5": spectrum_metadata_dict}
        )
        spectrum_signal.metadata.General.title = path.stem

        # Apply microscope parameters and calibration from the .jh5 metadata
        jh5_meta_as_dict = signal.original_metadata.as_dictionary()
        set_metadata_from_JH5metadata(spectrum_signal, jh5_meta_as_dict)
        set_calibration_from_JH5metadata(spectrum_signal, jh5_meta_as_dict)

        signal = spectrum_signal
        logger.info("Successfully loaded full EDS spectrum image.")

    except (FileNotFoundError, ValueError) as e:
        logger.debug(f"No spectrum image found or load failed: {e}")
    except Exception as e:
        logger.error(
            f"Unexpected error while loading spectrum image: {e}", exc_info=True
        )

    return signal
