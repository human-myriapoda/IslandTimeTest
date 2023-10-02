"""
This module allows us to retrieve the currently available data about a given island and print it.
It opens the file containing the dictionary `info_{island}_{country}.data`.
If no information is available yet, the code suggests to run `pre_timeseries_steps.py`.
TODO: last run date
TODO: overwrite

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import pre_timeseries_steps, timeseries_disasters_EMDAT, timeseries_environmental_climate_indices, \
                     timeseries_environmental_insitu_PSMSL, timeseries_environmental_insitu_PSLGM, \
                     timeseries_environmental_remotesensing_ERA5, timeseries_environmental_remotesensing_NOAA_CRW, \
                     timeseries_environmental_remotesensing_PMLV2, timeseries_socioeconomics_remotesensing_nighttimelight, add_ecological_coastal_units, get_OpenStreetMap_data, timeseries_environmental_remotesensing_sea_level_anomaly, \
                     timeseries_environmental_remotesensing_wave_energy
from mpetools.archives import timeseries_socioeconomics_environmental_WHO
from mpetools.archives_august_2023 import timeseries_socioeconomics_WorldBank
                     

def run_all_functions(island, country, method='new', overwrite=False):

    _ = pre_timeseries_steps.PreTimeSeries(island, country, method=method, overwrite=overwrite).main()
    _ = timeseries_disasters_EMDAT.TimeSeriesDisasters(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_climate_indices.TimeSeriesClimateIndices(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_insitu_PSMSL.TimeSeriesPSMSL(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_insitu_PSLGM.TimeSeriesPSLGM(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_remotesensing_ERA5.TimeSeriesERA5(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_remotesensing_NOAA_CRW.TimeSeriesNOAACRW(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_remotesensing_PMLV2.TimeSeriesPMLV2(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_remotesensing_sea_level_anomaly.TimeSeriesSeaLevelAnomaly(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_environmental_remotesensing_wave_energy.TimeSeriesWaveEnergy(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_socioeconomics_WorldBank.TimeSeriesWorldBank(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = timeseries_socioeconomics_remotesensing_nighttimelight.TimeSeriesNighttimeLight(island, country, verbose_init=False, overwrite=overwrite).main()
    _ = add_ecological_coastal_units.AddEcologicalCoastalUnits(island, country, verbose_init=False, overwrite=overwrite).main()
    island_info = timeseries_socioeconomics_environmental_WHO.TimeSeriesWHO(island, country, verbose_init=False, overwrite=overwrite).main()

    # = get_OpenStreetMap_data.getOpenStreetMap(island, country, verbose_init=False)

    return island_info