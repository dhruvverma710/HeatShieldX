# Status Labels
OBSERVED = "OBSERVED"
ESTIMATED = "ESTIMATED"
MODELLED = "MODELLED"
INTERPOLATED = "INTERPOLATED"

def attach_provenance(df, source_type, source_reference, observed_or_estimated, assumptions_version, computation_mode):
    """
    Attaches provenance fields to a DataFrame or GeoDataFrame.
    """
    df['prov_source_type'] = source_type
    df['prov_source_reference'] = source_reference
    df['prov_obs_est'] = observed_or_estimated
    df['prov_assumptions_version'] = assumptions_version
    df['prov_computation_mode'] = computation_mode
    return df
