from libra_toolbox.tritium.model import ureg, Model, quantity_to_activity
import numpy as np
import json
from libra_toolbox.tritium.lsc_measurements import (
    LIBRARun,
    LSCFileReader,
    GasStream,
    LSCSample,
    LIBRASample,
)

from datetime import datetime

def load_run_data():
    global OV_stream, IV_stream, baby_model, processed_data, replacement_times_top, replacement_times_walls

    data_folder = "../../data/tritium_detection_run1"

    # read files
    file_reader_1 = LSCFileReader(
        f"{data_folder}/1L_OV_IV_1-0-X_IV_1-1-X.csv",
        vial_labels=[
            "OV-1-0-1",
            "OV-1-0-2",
            "OV-1-0-3",
            "OV-1-0-4",
            None,
            "IV-1-0-1",
            "IV-1-0-2",
            "IV-1-0-3",
            "IV-1-0-4",
            None,
            "IV-1-1-1",
            "IV-1-1-2",
            "IV-1-1-3",
            "IV-1-1-4",
        ],
    )
    file_reader_1.read_file()

    file_reader_2 = LSCFileReader(
        f"{data_folder}/1L_IV_1-2-X.csv",
        vial_labels=[
            "IV-1-2-1",
            "IV-1-2-2",
            "IV-1-2-3",
            "IV-1-2-4",
        ],
    )
    file_reader_2.read_file()

    file_reader_3 = LSCFileReader(
        f"{data_folder}/IV_1-3-X_BL-1.csv",
        vial_labels=[
            "IV-BL-1",
            None,
            "IV-1-3-1",
            "IV-1-3-2",  # probably has a statistic issue
            "IV-1-3-3",
            "IV-1-3-4",
        ],
    )
    file_reader_3.read_file()

    file_reader_4 = LSCFileReader(
        f"{data_folder}/Report1.csv",
        vial_labels=[
            "BL-1_count_4",
            None,
            "IV-1-4-1",
            "IV-1-4-2",
            "IV-1-4-3",
            "IV-1-4-4",
            None,
            "OV-1-1-1",
            "OV-1-1-2",
            "OV-1-1-3",
            "OV-1-1-4",
            None,
            "IV-1-3-2 (repeat)",
        ],
    )
    file_reader_4.read_file()


    file_reader_OV_1_recount = LSCFileReader(
        f"{data_folder}/OV-1-1_with_avg.csv",
        vial_labels=[
            "BL-1_1",
            "BL-1_2",
            "BL-1_3",
            "BL-1_4",
            "BL-1_5",
            "BL-1_6",
            "BL-1_avg",
            None,
            "OV-1-1-1_1",
            "OV-1-1-1_2",
            "OV-1-1-1_3",
            "OV-1-1-1_4",
            "OV-1-1-1_5",
            "OV-1-1-1_6",
            "OV-1-1-1_avg",
            "OV-1-1-2_1",
            "OV-1-1-2_2",
            "OV-1-1-2_3",
            "OV-1-1-2_4",
            "OV-1-1-2_5",
            "OV-1-1-2_6",
            "OV-1-1-2_avg",
            "OV-1-1-3_1",
            "OV-1-1-3_2",
            "OV-1-1-3_3",
            "OV-1-1-3_4",
            "OV-1-1-3_5",
            "OV-1-1-3_6",
            "OV-1-1-3_avg",
            "OV-1-1-4_1",
            "OV-1-1-4_2",
            "OV-1-1-4_3",
            "OV-1-1-4_4",
            "OV-1-1-4_5",
            "OV-1-1-4_6",
            "OV-1-1-4_avg",
        ],
    )
    file_reader_OV_1_recount.read_file()

    file_reader_5 = LSCFileReader(
        f"{data_folder}/1L_BL-1_IV-1-5_OV-1-2.csv",
        vial_labels=[
            "1L-BL-1",
            None,
            "IV-1-5-1",
            "IV-1-5-2",
            "IV-1-5-3",
            "IV-1-5-4",
            None,
            "OV-1-2-1",
            "OV-1-2-2",
            "OV-1-2-3",
            "OV-1-2-4",
        ],
    )
    file_reader_5.read_file()


    file_reader_7 = LSCFileReader(
        f"{data_folder}/1L_BL-1_IV-1-6_OV-1-3.csv",
        vial_labels=[
            "1L-BL-1",
            None,
            "IV 1-6-1",
            "IV 1-6-2",
            "IV 1-6-3",
            "IV 1-6-4",
            None,
            "OV 1-3-1",
            "OV 1-3-2",
            "OV 1-3-3",
            "OV 1-3-4",
        ],
    )
    file_reader_7.read_file()

    file_reader_8 = LSCFileReader(
        f"{data_folder}/1L_BL-1_IV-1-7_OV-1-4.csv",
        vial_labels=[
            "1L-BL-1",
            None,
            "IV 1-7-1",
            "IV 1-7-2",
            "IV 1-7-3",
            "IV 1-7-4",
            None,
            "OV 1-4-1",
            "OV 1-4-2",
            "OV 1-4-3",
            "OV 1-4-4",
        ],
    )
    file_reader_8.read_file()

    # NOTE: 12/10/2024 This count is still ongoing, the OV 1-6 samples are not yet counted, but for all practical purposes we can assume they're zeros. Other than that it all looks good to me.
    file_reader_9 = LSCFileReader(
        f"{data_folder}/1L_BL-1_IV-1-8_OV-1-5_IV-1-9_OV-1-6.csv",
        vial_labels=[
            "1L-BL-1",
            None,
            "IV 1-8-1",
            "IV 1-8-2",
            "IV 1-8-3",
            "IV 1-8-4",
            None,
            "OV 1-5-1",
            "OV 1-5-2",
            "OV 1-5-3",
            "OV 1-5-4",
            None,
            "IV 1-9-1",
            "IV 1-9-2",
            "IV 1-9-3",
            "IV 1-9-4",
            None,
            "OV 1-6-1",
            "OV 1-6-2",
            "OV 1-6-3",
            "OV 1-6-4",
        ],
    )
    file_reader_9.read_file()


    # Make samples
    with open("../../data/general_run1.json", "r") as f:
        general_data = json.load(f)


    time_sample_0_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-0-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_0_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_1, label)
            for label in ["IV-1-0-1", "IV-1-0-2", "IV-1-0-3", "IV-1-0-4"]
        ],
        time=time_sample_0_IV,
    )

    time_sample_1_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-1-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )

    sample_1_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_1, label)
            for label in ["IV-1-1-1", "IV-1-1-2", "IV-1-1-3", "IV-1-1-4"]
        ],
        time=time_sample_1_IV,
    )

    time_sample_2_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-2-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_2_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_2, label)
            for label in ["IV-1-2-1", "IV-1-2-2", "IV-1-2-3", "IV-1-2-4"]
        ],
        time=time_sample_2_IV,
    )

    time_sample_3_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-3-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_3_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_3, "IV-1-3-1"),
            LSCSample.from_file(
                file_reader_4, "IV-1-3-2 (repeat)"
            ),  # the first one has a statistic issue
            LSCSample.from_file(file_reader_3, "IV-1-3-3"),
            LSCSample.from_file(file_reader_3, "IV-1-3-4"),
        ],
        time=time_sample_3_IV,
    )
    blank_sample_3_IV = LSCSample.from_file(file_reader_3, "IV-BL-1")

    time_sample_4_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-4-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_4_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_4, label)
            for label in ["IV-1-4-1", "IV-1-4-2", "IV-1-4-3", "IV-1-4-4"]
        ],
        time=time_sample_4_IV,
    )
    blank_sample_4 = LSCSample.from_file(file_reader_4, "BL-1_count_4")

    time_sample_1_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-1-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_1_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_OV_1_recount, label)
            for label in ["OV-1-1-1_avg", "OV-1-1-2_avg", "OV-1-1-3_avg", "OV-1-1-4_avg"]
        ],
        time=time_sample_1_OV,
    )
    blank_sample_1_OV = LSCSample.from_file(file_reader_OV_1_recount, "BL-1_avg")

    time_sample_5_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-5-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_5_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_5, label)
            for label in ["IV-1-5-1", "IV-1-5-2", "IV-1-5-3", "IV-1-5-4"]
        ],
        time=time_sample_5_IV,
    )
    sample_5_IV_background = LSCSample.from_file(file_reader_5, "1L-BL-1")
    sample_2_OV_background = sample_5_IV_background

    time_sample_2_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-2-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_2_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_5, label)
            for label in ["OV-1-2-1", "OV-1-2-2", "OV-1-2-3", "OV-1-2-4"]
        ],
        time=time_sample_2_OV,
    )

    time_sample_6_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-6-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_6_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_7, label)
            for label in ["IV 1-6-1", "IV 1-6-2", "IV 1-6-3", "IV 1-6-4"]
        ],
        time=time_sample_6_IV,
    )

    time_sample_3_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-3-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_3_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_7, label)
            for label in ["OV 1-3-1", "OV 1-3-2", "OV 1-3-3", "OV 1-3-4"]
        ],
        time=time_sample_3_OV,
    )

    background_file_7 = LSCSample.from_file(file_reader_7, "1L-BL-1")

    time_sample_7_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-7-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_7_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_8, label)
            for label in ["IV 1-7-1", "IV 1-7-2", "IV 1-7-3", "IV 1-7-4"]
        ],
        time=time_sample_7_IV,
    )

    time_sample_8_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-8-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_8_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_9, label)
            for label in ["IV 1-8-1", "IV 1-8-2", "IV 1-8-3", "IV 1-8-4"]
        ],
        time=time_sample_8_IV,
    )

    background_file_9 = LSCSample.from_file(file_reader_9, "1L-BL-1")

    time_sample_4_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-4-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_4_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_8, label)
            for label in ["OV 1-4-1", "OV 1-4-2", "OV 1-4-3", "OV 1-4-4"]
        ],
        time=time_sample_4_OV,
    )

    background_file_8 = LSCSample.from_file(file_reader_8, "1L-BL-1")

    time_sample_9_IV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["IV"]["1-9-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_9_IV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_9, label)
            for label in ["IV 1-9-1", "IV 1-9-2", "IV 1-9-3", "IV 1-9-4"]
        ],
        time=time_sample_9_IV,
    )

    time_sample_5_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-5-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_5_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_9, label)
            for label in ["OV 1-5-1", "OV 1-5-2", "OV 1-5-3", "OV 1-5-4"]
        ],
        time=time_sample_5_OV,
    )

    time_sample_6_OV = datetime.strptime(
        general_data["timestamps"]["lsc_sample_times"]["OV"]["1-6-x"]["actual"],
        "%m/%d/%Y %H:%M",
    )
    sample_6_OV = LIBRASample(
        samples=[
            LSCSample.from_file(file_reader_9, label)
            for label in ["OV 1-6-1", "OV 1-6-2", "OV 1-6-3", "OV 1-6-4"]
        ],
        time=time_sample_6_OV,
    )

    # Make streams

    # read start time from general.json
    all_start_times = []
    for generator in general_data["generators"]:
        if generator["enabled"] is False:
            continue
        for irradiation_period in generator["periods"]:
            start_time = datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
            all_start_times.append(start_time)
    start_time = min(all_start_times)

    IV_stream = GasStream(
        [
            sample_1_IV,
            sample_2_IV,
            sample_3_IV,
            sample_4_IV,
            sample_5_IV,
            sample_6_IV,
            sample_7_IV,
            sample_8_IV,
            sample_9_IV,
        ],
        start_time=start_time,
    )
    OV_stream = GasStream(
        [sample_1_OV, sample_2_OV, sample_3_OV, sample_4_OV, sample_5_OV, sample_6_OV],
        start_time=start_time,
    )

    gas_streams = {
        "IV": IV_stream,
        "OV": OV_stream,
    }

    # substract background
    for sample in [sample_1_IV, sample_2_IV]:
        sample.substract_background(
            background_sample=LSCSample(activity=0.320 * ureg.Bq, name="background")
        )  # TODO don't have a real background here

    sample_3_IV.substract_background(background_sample=blank_sample_3_IV)
    sample_4_IV.substract_background(background_sample=blank_sample_4)
    sample_5_IV.substract_background(background_sample=sample_5_IV_background)
    sample_6_IV.substract_background(background_sample=background_file_7)
    sample_7_IV.substract_background(background_sample=background_file_8)
    sample_8_IV.substract_background(background_sample=background_file_9)
    sample_9_IV.substract_background(background_sample=background_file_9)
    sample_1_OV.substract_background(background_sample=blank_sample_1_OV)
    sample_2_OV.substract_background(background_sample=sample_2_OV_background)
    sample_3_OV.substract_background(background_sample=background_file_7)
    sample_4_OV.substract_background(background_sample=background_file_8)
    sample_5_OV.substract_background(background_sample=background_file_9)
    sample_6_OV.substract_background(background_sample=background_file_9)

    # create run
    run = LIBRARun(streams=[IV_stream, OV_stream], start_time=start_time)

    # check that background is always substracted
    for stream in run.streams:
        for sample in stream.samples:
            for lsc_vial in sample.samples:
                assert (
                    lsc_vial.background_substracted
                ), f"Background not substracted for {sample}"


    replacement_times_top = sorted(IV_stream.relative_times_as_pint)
    replacement_times_walls = sorted(OV_stream.relative_times_as_pint)


    # tritium model

    baby_diameter = 14 * ureg.cm  # TODO confirm with CAD
    baby_radius = 0.5 * baby_diameter
    baby_volume = 1 * ureg.L
    baby_cross_section = np.pi * baby_radius**2
    baby_height = baby_volume / baby_cross_section

    # read irradiation times from general.json

    irradiations = []
    for generator in general_data["generators"]:
        if generator["enabled"] is False:
            continue
        for irradiation_period in generator["periods"]:
            irr_start_time = (
                datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
                - start_time
            )
            irr_stop_time = (
                datetime.strptime(irradiation_period["end"], "%m/%d/%Y %H:%M") - start_time
            )
            irr_start_time = irr_start_time.total_seconds() * ureg.second
            irr_stop_time = irr_stop_time.total_seconds() * ureg.second
            irradiations.append([irr_start_time, irr_stop_time])

    # Neutron rate
    # taking from https://github.com/LIBRA-project/libra-toolbox/pull/47/files#diff-6d8628c84de8f0814868feb8a67e92d36d4abea51105d48f46df380d7cda4d5a
    # TODO replace by robust method once https://github.com/LIBRA-project/libra-toolbox/pull/47 is merged
    neutron_rate_relative_uncertainty = 0.089
    neutron_rate = np.mean([9.426e7, 8.002e7, 1.001e8]) * ureg.neutron * ureg.s**-1

    # TBR from OpenMC

    from pathlib import Path

    filename = "../neutron/statepoint.100.h5"
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist, run OpenMC first")

    import openmc

    sp = openmc.StatePoint(filename)
    tally_df = sp.get_tally(name="TBR").get_pandas_dataframe()
    calculated_TBR = tally_df["mean"].iloc[0] * ureg.particle * ureg.neutron**-1
    calculated_TBR_std_dev = (
        tally_df["std. dev."].iloc[0] * ureg.particle * ureg.neutron**-1
    )

    # TBR from measurements

    total_irradiation_time = sum([irr[1] - irr[0] for irr in irradiations])

    T_consumed = neutron_rate * total_irradiation_time
    T_produced = sum(
        [stream.get_cumulative_activity("total")[-1] for stream in run.streams]
    )

    measured_TBR = (T_produced / quantity_to_activity(T_consumed)).to(
        ureg.particle * ureg.neutron**-1
    )

    optimised_ratio = 1.7e-2
    k_top = 8.9e-8 * ureg.m * ureg.s**-1
    k_wall = optimised_ratio * k_top


    baby_model = Model(
        radius=baby_radius,
        height=baby_height,
        TBR=measured_TBR,
        neutron_rate=neutron_rate,
        irradiations=irradiations,
        k_top=k_top,
        k_wall=k_wall,
    )


    # store processed data
    processed_data = {
        "modelled_baby_radius": {
            "value": baby_radius.magnitude,
            "unit": str(baby_radius.units),
        },
        "modelled_baby_height": {
            "value": baby_height.magnitude,
            "unit": str(baby_height.units),
        },
        "irradiations": [
            {
                "start_time": {
                    "value": irr[0].magnitude,
                    "unit": str(irr[0].units),
                },
                "stop_time": {
                    "value": irr[1].magnitude,
                    "unit": str(irr[1].units),
                },
            }
            for irr in irradiations
        ],
        "neutron_rate_used_in_model": {
            "value": baby_model.neutron_rate.magnitude,
            "unit": str(baby_model.neutron_rate.units),
        },
        # TODO remove this and have it done by the neutron analysis once https://github.com/LIBRA-project/libra-toolbox/pull/47 is merged
        "measured_neutron_rate": {
            "value": neutron_rate.magnitude,
            "unit": str(neutron_rate.units),
        },
        "measured_TBR": {
            "value": measured_TBR.magnitude,
            "unit": str(measured_TBR.units),
        },
        "TBR_used_in_model": {
            "value": baby_model.TBR.magnitude,
            "unit": str(baby_model.TBR.units),
        },
        "k_top": {
            "value": baby_model.k_top.magnitude,
            "unit": str(baby_model.k_top.units),
        },
        "k_wall": {
            "value": baby_model.k_wall.magnitude,
            "unit": str(baby_model.k_wall.units),
        },
        "cumulative_tritium_release": {
            label: {
                **{
                    form: {
                        "value": gas_stream.get_cumulative_activity(
                            form
                        ).magnitude.tolist(),
                        "unit": str(gas_stream.get_cumulative_activity(form).units),
                    }
                    for form in ["total", "soluble", "insoluble"]
                },
                "sampling_times": {
                    "value": gas_stream.relative_times_as_pint.magnitude.tolist(),
                    "unit": str(gas_stream.relative_times_as_pint.units),
                },
            }
            for label, gas_stream in gas_streams.items()
        },
    }

    # check if the file exists and load it

    processed_data_file = "../../data/processed_data_run1.json"

    try:
        with open(processed_data_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        print(f"Processed data file not found, creating it in {processed_data_file}")
        existing_data = {}

    existing_data.update(processed_data)

    with open(processed_data_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Processed data stored in {processed_data_file}")

load_run_data()
