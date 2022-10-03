from pandas import concat

def to_single_stage_performance_evaluation(
        to_metrics_in,
        stage_name,
        idc_in
):
    to_metrics_in_idc = to_metrics_in.loc[:, idc_in]
    to_metrics_in = to_metrics_in.drop(columns=[idc_in])
    to_metrics_in[to_metrics_in != stage_name] = 'Any'

    to_metrics = concat([to_metrics_in_idc, to_metrics_in], axis=1)

    return to_metrics
