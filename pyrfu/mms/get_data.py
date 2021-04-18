# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.


from astropy.time import Time

from .split_vs import split_vs
from .list_files import list_files
from .get_dist import get_dist
from .get_ts import get_ts
from ..pyrf import ts_append, dist_append


def get_data(var_str, tint, mms_id, verbose=True, data_path=""):
    """Load a variable. var_str must be in var (see below)

    * EPHEMERIS :
        "R_gse", "R_gsm"

    * FGM :
        "B_gsm_fgm_srvy_l2", "B_gsm_fgm_brst_l2", "B_gse_fgm_srvy_l2",
        "B_gse_fgm_brst_l2", "B_bcs_fgm_srvy_l2", "B_bcs_fgm_brst_l2",
        "B_dmpa_fgm_srvy_l2", "B_dmpa_fgm_brst_l2"

    * DFG & AFG :
        "B_gsm_dfg_srvy_l2pre", "B_gse_dfg_srvy_l2pre", "B_dmpa_dfg_srvy_l2pre",
        "B_bcs_dfg_srvy_l2pre", "B_gsm_afg_srvy_l2pre", "B_gse_afg_srvy_l2pre",
        "B_dmpa_afg_srvy_l2pre", "B_bcs_afg_srvy_l2pre"

    * SCM :
        "B_gse_scm_brst_l2"

    * EDP :
        "Phase_edp_fast_l2a", "Phase_edp_slow_l2a", "Sdev12_edp_slow_l2a",
        "Sdev34_edp_slow_l2a", "Sdev12_edp_fast_l2a", "Sdev34_edp_fast_l2a",
        "E_dsl_edp_brst_l2", "E_dsl_edp_fast_l2", "E_dsl_edp_brst_ql",
        "E_dsl_edp_fast_ql", "E_dsl_edp_slow_l2", "E_gse_edp_brst_l2",
        "E_gse_edp_fast_l2", "E_gse_edp_slow_l2", "E2d_dsl_edp_brst_l2pre",
        "E2d_dsl_edp_fast_l2pre", "E2d_dsl_edp_brst_ql", "E2d_dsl_edp_fast_ql",
        "E2d_dsl_edp_l2pre", "E2d_dsl_edp_fast_l2pre", "E2d_dsl_edp_brst_l2pre",
        "E_dsl_edp_l2pre", "E_dsl_edp_fast_l2pre", "E_dsl_edp_brst_l2pre",
        "E_dsl_edp_slow_l2pre", "E_ssc_edp_brst_l2a", "E_ssc_edp_fast_l2a",
        "E_ssc_edp_slow_l2a", "V_edp_fast_sitl", "V_edp_slow_sitl",
        "V_edp_slow_l2", "V_edp_fast_l2", "V_edp_brst_l2"

    * FPI Ions :
        "Vi_dbcs_fpi_brst_l2", "Vi_dbcs_fpi_fast_l2", "Vi_dbcs_fpi_l2",
        "Vi_gse_fpi_ql", "Vi_gse_fpi_fast_ql", "Vi_dbcs_fpi_fast_ql",
        "Vi_gse_fpi_fast_l2", "Vi_gse_fpi_brst_l2", "partVi_gse_fpi_brst_l2",
        "Ni_fpi_brst_l2", "partNi_fpi_brst_l2", "Ni_fpi_brst",
        "Ni_fpi_fast_l2", "Ni_fpi_ql", "DEFi_fpi_fast_ql",
        "DEFi_fpi_fast_l2", "Tperpi_fpi_brst_l2", "Tparai_fpi_brst_l2",
        "partTperpi_fpi_brst_l2", "partTparai_fpi_brst_l2", "Ti_dbcs_fpi_brst_l2",
        "Ti_dbcs_fpi_brst", "Ti_dbcs_fpi_fast_l2", "Ti_gse_fpi_ql",
        "Ti_dbcs_fpi_ql", "Ti_gse_fpi_brst_l2", "Pi_dbcs_fpi_brst_l2",
        "Pi_dbcs_fpi_brst", "Pi_dbcs_fpi_fast_l2", "Pi_gse_fpi_ql",
        "Pi_gse_fpi_brst_l2"

    * FPI Electrons :
        "Ve_dbcs_fpi_brst_l2", "Ve_dbcs_fpi_brst", "Ve_dbcs_fpi_ql",
        "Ve_dbcs_fpi_fast_l2", "Ve_gse_fpi_ql", "Ve_gse_fpi_fast_l2",
        "Ve_gse_fpi_brst_l2", "partVe_gse_fpi_brst_l2", "DEFe_fpi_fast_ql",
        "DEFe_fpi_fast_l2", "Ne_fpi_brst_l2", "partNe_fpi_brst_l2",
        "Ne_fpi_brst", "Ne_fpi_fast_l2", "Ne_fpi_ql",
        "Tperpe_fpi_brst_l2", "Tparae_fpi_brst_l2", "partTperpe_fpi_brst_l2",
        "partTparae_fpi_brst_l2", "Te_dbcs_fpi_brst_l2", "Te_dbcs_fpi_brst",
        "Te_dbcs_fpi_fast_l2", "Te_gse_fpi_ql", "Te_dbcs_fpi_ql",
        "Te_gse_fpi_brst_l2", "Pe_dbcs_fpi_brst_l2", "Pe_dbcs_fpi_brst",
        "Pe_dbcs_fpi_fast_l2", "Pe_gse_fpi_ql", "Pe_gse_fpi_brst_l2",

    * HPCA :
        "Nhplus_hpca_srvy_l2", "Nheplus_hpca_srvy_l2", "Nheplusplus_hpca_srvy_l2",
        "Noplus_hpca_srvy_l2", "Tshplus_hpca_srvy_l2", "Tsheplus_hpca_srvy_l2",
        "Tsheplusplus_hpca_srvy_l2", "Tsoplus_hpca_srvy_l2", "Vhplus_dbcs_hpca_srvy_l2",
        "Vheplus_dbcs_hpca_srvy_l2", "Vheplusplus_dbcs_hpca_srvy_l2", "Voplus_dbcs_hpca_srvy_l2",
        "Phplus_dbcs_hpca_srvy_l2", "Pheplus_dbcs_hpca_srvy_l2", "Pheplusplus_dbcs_hpca_srvy_l2",
        "Poplus_dbcs_hpca_srvy_l2", "Thplus_dbcs_hpca_srvy_l2", "Theplus_dbcs_hpca_srvy_l2",
        "Theplusplus_dbcs_hpca_srvy_l2", "Toplus_dbcs_hpca_srvy_l2", "Vhplus_gsm_hpca_srvy_l2",
        "Vheplus_gsm_hpca_srvy_l2", "Vheplusplus_gsm_hpca_srvy_l2", "Voplus_gsm_hpca_srvy_l2",
        "Nhplus_hpca_brst_l2", "Nheplus_hpca_brst_l2", "Nheplusplus_hpca_brst_l2",
        "Noplus_hpca_brst_l2", "Tshplus_hpca_brst_l2", "Tsheplus_hpca_brst_l2",
        "Tsheplusplus_hpca_brst_l2", "Tsoplus_hpca_brst_l2", "Vhplus_dbcs_hpca_brst_l2",
        "Vheplus_dbcs_hpca_brst_l2", "Vheplusplus_dbcs_hpca_brst_l2", "Voplus_dbcs_hpca_brst_l2",
        "Phplus_dbcs_hpca_brst_l2", "Pheplus_dbcs_hpca_brst_l2", "Pheplusplus_dbcs_hpca_brst_l2",
        "Poplus_dbcs_hpca_brst_l2", "Thplus_dbcs_hpca_brst_l2", "Theplus_dbcs_hpca_brst_l2",
        "Theplusplus_dbcs_hpca_brst_l2", "Toplus_dbcs_hpca_brst_l2", "Vhplus_gsm_hpca_brst_l2",
        "Vheplus_gsm_hpca_brst_l2", "Vheplusplus_gsm_hpca_brst_l2", "Voplus_gsm_hpca_brst_l2",
        "Phplus_gsm_hpca_brst_l2", "Pheplus_gsm_hpca_brst_l2", "Pheplusplus_gsm_hpca_brst_l2",
        "Poplus_gsm_hpca_brst_l2", "Thplus_gsm_hpca_brst_l2", "Theplus_gsm_hpca_brst_l2",
        "Theplusplus_gsm_hpca_brst_l2", "Toplus_gsm_hpca_brst_l2"

    Parameters
    ----------
    var_str : str
        Key of the target variable (see above).

    tint : list of str
        Time interval.

    mms_id : str or int
        Index of the target spacecraft.

    verbose : bool, optional
        Set to True to follow the loading. Default is True.

    data_path : str, optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the target variable of measured by the target spacecraft over the selected
        time interval.

    See also
    --------
    pyrfu.mms.get_ts : Read time series.
    pyrfu.mms.get_dist : Read velocity distribution function.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of MMS spacecraft

    >>> ic = 1

    Load magnetic field from FGM

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint_brst, ic)

    """

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)

    # Translate short names to names readable for splitVs
    if var_str == "dfg_ql_srvy":
        var_str = "B_dmpa_dfg_srvy_ql"

    elif var_str == "afg_ql_srvy":
        var_str = "B_dmpa_afg_srvy_ql"

    elif var_str == "Nhplus_hpca_sitl":
        var_str = "Nhplus_hpca_srvy_sitl"

    elif var_str.lower() in ["r_gse", "r_gsm", "v_gse", "v_gsm"]:
        var_str = "_".join([var_str, "mec", "srvy", "l2"])

    var = split_vs(var_str)

    mms_id_str = "mms{}".format(mms_id)

    var["dtype"] = None

    vdf_flag = False

    if var["inst"] == "mec":
        cdf_name = "_".join([mms_id_str, "mec", var["param"].lower(), var["cs"]])

        var["dtype"] = "epht89d"

    elif var["inst"] == "fsm":
        cdf_name = "_".join([mms_id_str, var["inst"], "b", var["cs"], var["tmmode"], var["lev"]])

        var["dtype"] = "8khz"

    elif var["inst"] in ["fgm", "dfg", "afg"]:
        if var["lev"] == "l2":
            cdf_name = "_".join(
                [mms_id_str, var["inst"], "b", var["cs"], var["tmmode"], var["lev"]])

        elif var["lev"] == "l2pre":
            cdf_name = "_".join(
                [mms_id_str, var["inst"], "b", var["cs"], var["tmmode"], var["lev"]])

        elif var["lev"] == "ql":
            if var["cs"] == "dmpa":
                cdf_name = "_".join([mms_id_str, var["inst"], var["tmmode"], var["cs"]])
            elif var["cs"] == "gsm":
                cdf_name = "_".join([mms_id_str, var["inst"], var["tmmode"], var["cs"], "dmpa"])
            else:
                raise ValueError("Invalid coordinates")
        else:
            raise InterruptedError("Should not be here")

    elif var["inst"] == "fpi":
        # get specie
        if var["param"][-1] == "i":
            sensor = "dis"

        elif var["param"][-1] == "e":
            sensor = "des"

        else:
            raise ValueError("invalid specie")

        if var["lev"] in ["l2", "l2pre", "l1b"]:
            if var["param"][:2].lower() == "pd":
                vdf_flag = True

                var["dtype"] = "{}-dist".format(sensor)

            else:
                if len(var["param"]) > 4:
                    if var["param"][:4] == "part":
                        var["dtype"] = "{}-partmoms".format(sensor)

                    else:
                        var["dtype"] = "{}-moms".format(sensor)

                else:
                    var["dtype"] = "{}-moms".format(sensor)

        elif var["lev"] == "ql":
            var["dtype"] = sensor

        else:
            raise InterruptedError("Should not be here")

        if var["param"].lower() in ["pdi", "pde", "pderri", "pderre"]:
            if var["param"][:-1].lower() == "pd":
                cdf_name = "_".join([mms_id_str, sensor, "dist", var["tmmode"]])

            elif len(var["param"]) == 6 and var["param"][:-1].lower() == "pderr":
                cdf_name = "_".join([mms_id_str, sensor, "disterr", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Number density
        elif var["param"].lower() in ["ne", "ni"]:
            if var["lev"] in ["l2", "l2pre"]:
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity", var["tmmode"]])

            elif var["lev"] == "l1b":
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity"])

            elif var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity", "fast"])

            elif var["lev"] == "sitl":
                cdf_name = "_".join([mms_id_str, "fpi", sensor.upper(), "numberDensity"])

            else:
                raise InterruptedError("Should not be here")

        # Number density
        elif var["param"].lower() in ["nbge", "nbgi"]:
            if var["lev"] in ["l2", "l2pre"]:
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity_bg", var["tmmode"]])

            elif var["lev"] == "l1b":
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity_bg"])

            elif var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity_bg", "fast"])

            else:
                raise InterruptedError("Should not be here")

        # Partial number density
        elif var["param"].lower() in ["partni", "partne"]:
            if var["lev"] == "l2":
                # only for l2 data
                cdf_name = "_".join([mms_id_str, sensor, "numberdensity", "part", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Energy flux omni
        elif var["param"].lower() in ["defi", "defe"]:
            if var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "energyspectr", "omni", "fast"])

            elif var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "energyspectr", "omni", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Energy flux omni
        elif var["param"].lower() in ["defbgi", "defbge"]:
            if var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "spectr_bg", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Energies
        elif var["param"].lower() in ["energyi", "energye"]:
            if var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "energy", "fast"])

            elif var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "energy", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Parallel and perpendicular temperatures
        elif var["param"].lower() in ["tparai", "tparae", "tperpi", "tperpe"]:
            # Field Aligned component (either para or perp)
            tmp_fac = var["param"][1:5]

            if var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "temp{}".format(tmp_fac), var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Partial moments parallel and perpandiculat temperatures
        elif var["param"].lower() in ["parttparai", "parttparae", "parttperpi", "parttperpe"]:
            # Field Aligned component (either para or perp)
            tmp_fac = var["param"][5:9]

            if var["lev"] == "l2":
                cdf_name = "_".join(
                    [mms_id_str, sensor, "temp{}".format(tmp_fac), "part", var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Temperature and pressure tensors
        elif var["param"].lower() in ["ti", "te", "pi", "pe"]:
            if var["param"][0].lower() == "t":
                mom_type = "temptensor"  # temperature

            elif var["param"][0].lower() == "p":
                mom_type = "prestensor"  # pressure

            else:
                raise InterruptedError("Should not be here")

            cdf_name = "_".join([mms_id_str, sensor, mom_type, var["cs"], var["tmmode"]])

        elif var["param"].lower() in ["pbgi", "pbge"]:
            mom_type = "pres_bg"

            cdf_name = "_".join([mms_id_str, sensor, mom_type, var["tmmode"]])

        # Partial temperature and pressure tensors
        elif var["param"].lower() in ["partti", "partte", "partpi", "partpe"]:
            if var["param"][4] == "T":
                mom_type = "temptensor"  # temperature

            elif var["param"][4] == "P":
                mom_type = "prestensor"  # pressure

            else:
                raise InterruptedError("Should not be here")

            cdf_name = "_".join([mms_id_str, sensor, mom_type, "part", var["cs"], var["tmmode"]])

        # spintone
        elif var["param"].lower() in ["sti", "ste"]:
            if var["lev"] in ["l2", "l2pre"]:
                cdf_name = "_".join(
                    [mms_id_str, sensor, "bulkv_spintone", var["cs"], var["tmmode"]])

            elif var["lev"] == "ql":
                cdf_name = "_".join(
                    [mms_id_str, sensor, "bulkv_spintone", var["cs"], var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Bulk velocities
        elif var["param"].lower() in ["vi", "ve"]:
            if var["lev"] in ["l2", "l2pre"]:
                cdf_name = "_".join([mms_id_str, sensor, "bulkv", var["cs"], var["tmmode"]])

            elif var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "bulkv", var["cs"], var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Error bulk velocities
        elif var["param"].lower() in ["errvi", "errve"]:
            if var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "bulkv", "err", var["tmmode"]])

            else:
                raise InterruptedError("Only l2 partmoms available now")

        # Partial bulk velocities
        elif var["param"].lower() in ["partvi", "partve"]:
            if var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "bulkv", "part", var["cs"], var["tmmode"]])

            else:
                raise InterruptedError("Only l2 partmoms available now")

        # Heat flux
        elif var["param"].lower() in ["qi", "qe"]:
            if var["lev"] in ["l2", "l2pre"]:
                cdf_name = "_".join([mms_id_str, sensor, "heatq", var["cs"], var["tmmode"]])

            elif var["lev"] == "ql":
                cdf_name = "_".join([mms_id_str, sensor, "heatq", var["cs"], var["tmmode"]])

            else:
                raise InterruptedError("Should not be here")

        # Heat flux error
        elif var["param"].lower() in ["errqi", "errqe"]:
            if var["lev"] in ["l2"]:
                cdf_name = "_".join([mms_id_str, sensor, "heatq", "err", var["tmmode"]])
            else:
                raise InterruptedError("Should not be here")

        elif var["param"].lower() in ["padlowene", "padmidene", "padhighene"]:
            # Energy range
            en_range = var["param"][3:-1]

            if var["lev"] == "l2":
                cdf_name = "_".join([mms_id_str, sensor, "pitchangdist", en_range, var["tmmode"]])

            else:
                raise InterruptedError("Only l2 partmoms available now")

        else:
            raise InterruptedError("Should not be here")

    # Hot Plasma Composition Analyser
    elif var["inst"] == "hpca":
        if var["param"][:3].lower() == "dpf":
            var["dtype"] = "ion"
            ion = var["param"].lower().strip("dpf")
            cdf_name = "{}_{}_{}_flux".format(mms_id_str, var["inst"], ion)
        elif var["param"].lower() == "saz":
            var["dtype"] = "ion"
            cdf_name = "{}_{}_start_azimuth".format(mms_id_str, var["inst"])
            # Number density
        elif var["param"][0].lower() == "n":
            var["dtype"] = "moments"
            ion = var["param"][1:]
            cdf_name = "_".join([mms_id_str, "hpca", ion, "number_density"])

            # Bulk velocity
        elif var["param"][0].lower() == "v":
            var["dtype"] = "moments"
            ion = var["param"][1:]
            cdf_name = "_".join([mms_id_str, "hpca", ion, "ion_bulk_velocity"])

            # Pressure tensor
        elif var["param"][0].lower() == "p":
            var["dtype"] = "moments"
            ion = var["param"][1:]
            cdf_name = "_".join([mms_id_str, "hpca", ion, "ion_pressure"])

            # Temperature tensor
        elif var["param"][0].lower() == "t":
            var["dtype"] = "moments"
            ion = var["param"][1:]
            cdf_name = "_".join([mms_id_str, "hpca", ion, "temperature_tensor"])

        else:
            raise ValueError("Unrecognized param")

        # Tensor (vector or matrix) add coordinate system to cdf_name
        if var["to"] > 0:
            if var["cs"].lower() == "gsm":
                cdf_name = "_".join([cdf_name, "GSM"])

            elif var["cs"].lower() == "dbcs":
                pass

            else:
                raise ValueError("invalid CS")

    # Search Coil Magnetometer
    elif var["inst"] == "scm":
        if var["lev"] != "l2":
            raise InterruptedError("not implemented yet")

        var["dtype"] = "scb"

        cdf_name = "_".join([mms_id_str, "scm", "acb", var["cs"], "scb", var["tmmode"], var["lev"]])

    elif var["inst"] == "dsp":
        if var["lev"] == "l2":
            if var["param"][0].lower() == "e":
                var["dtype"] = "{}psd".format(var["param"][0].lower())

                cdf_name = "_".join([mms_id_str, "dsp", var["dtype"], "omni"])

            elif var["param"][0].lower() == "b":
                var["dtype"] = "{}psd".format(var["param"][0].lower())

                cdf_name = "_".join(
                    [mms_id_str, "dsp", var["dtype"], "omni", var["tmmode"], var["lev"]])

            else:
                raise ValueError("Should not be here")
        else:
            raise ValueError("Should not be here")

    # Spin-plane Double mmsId instrument
    elif var["inst"] == "edp":
        if var["lev"] == "sitl":
            if var["param"].lower() == "e":
                param = "_".join(["dce_xyz", var["cs"]])

                var["dtype"] = "dce"

            elif var["param"].lower() == "e2d":
                param = "_".join(["dce_xyz", var["cs"]])

                var["dtype"] = "dce2d"

            elif var["param"].lowr() == "v":
                param = "scpot"

                var["dtype"] = "scpot"

            else:
                raise ValueError("Invalid param")

            cdf_name = "_".join([mms_id_str, "edp", param, var["tmmode"], var["lev"]])

        elif var["lev"] == "ql":
            if var["param"].lower() == "e":
                param = "_".join(["dce_xyz", var["cs"]])

                var["dtype"] = "dce"

            elif var["param"].lower() == "e2d":
                param = "_".join(["dce_xyz", var["cs"]])

                var["dtype"] = "dce2d"

            else:
                raise ValueError("Invalid param")

            cdf_name = "_".join([mms_id_str, "edp", param])

        elif var["lev"] == "l1b":
            if var["param"].lower() == "e":
                param = "dce_sensor"

                var["dtype"] = "dce"

            elif var["param"].lower() == "v":
                param = "dcv_sensor"

                var["dtype"] = "dce"

            else:
                raise ValueError("Invalid param")

            cdf_name = "_".join([mms_id_str, "edp", param])

        elif var["lev"] == "l2a":
            var["dtype"] = "dce2d"

            if var["param"].lower() == "phase":
                param = "phase"

            elif var["param"].lower() in ["es12", "es34"]:
                param = "espin_p{}".format(var["param"][2:4])

            elif var["param"].lower() == "adcoff":
                param = "adc_offset"

            elif var["param"].lower() in ["sdev12", "sdev34"]:
                param = "sdevfit_p{}".format(var["param"][4:6])

            elif var["param"].lower() == "e":
                param = "dce"

            else:
                raise ValueError("Invalid param")

            cdf_name = "_".join([mms_id_str, "edp", param, var["tmmode"], var["lev"]])

        else:
            if var["param"].lower() == "e":
                param = "dce_{}".format(var["cs"])

                var["dtype"] = "dce"

            elif var["param"].lower() == "epar":
                param = "dce_par_epar"

                var["dtype"] = "dce"

            elif var["param"].lower() == "e2d":
                param = "dce_{}".format(var["cs"])

                var["dtype"] = "dce2d"

            elif var["param"].lower() == "v":
                param = "scpot"

                var["dtype"] = "scpot"

            elif var["param"].lower() == "v6":
                var["dtype"] = "scpot"

                param = "dcv"

            else:
                raise ValueError("Invalid param")

            cdf_name = "_".join([mms_id_str, "edp", param, var["tmmode"], var["lev"]])

    else:
        raise ValueError("not implemented yet")

    files = list_files(tint, mms_id, var, data_path)

    if not files:
        raise ValueError("No files found. Make sure that the data_path is correct")

    if verbose:
        print(f"Loading {cdf_name}...")

    out = None

    for file in files:
        if vdf_flag:
            out = dist_append(out, get_dist(file, cdf_name, tint))

        else:
            out = ts_append(out, get_ts(file, cdf_name, tint))

    if out.time.data.dtype == "float64":
        out.time.data = Time(1e-9 * out.time.data, format="unix").datetime

    return out
