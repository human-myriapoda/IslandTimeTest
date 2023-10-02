import pytz

utc=pytz.UTC

# DataFrames 
df_to_match = np.array([df_MEI_c, df_sl_Gan, df_mean_2m_air_temperature, df_mean_sea_level_pressure, df_total_precipitation, df_sst, df_ce], dtype=object)
df_monthly = []

for df in df_to_match:
    name_column = df.columns[1]
    globals()['df_monthly_'+name_column] = df.groupby(pd.Grouper(key='datetime', freq='M'))[name_column].mean()

    if name_column == 'coastline_3':
        globals()['df_monthly_'+name_column] = globals()['df_monthly_'+name_column][(globals()['df_monthly_'+name_column].index > datetime.datetime(2014, 11, 12, tzinfo=utc))\
                                                                                    & (globals()['df_monthly_'+name_column].index < datetime.datetime(2018, 11, 12, tzinfo=utc))]

    else:
        globals()['df_monthly_'+name_column] = globals()['df_monthly_'+name_column][(globals()['df_monthly_'+name_column].index > datetime.datetime(2014, 11, 12)) \
                                                                                    & (globals()['df_monthly_'+name_column].index < datetime.datetime(2018, 11, 12))]

    plt.plot(globals()['df_monthly_'+name_column].index, globals()['df_monthly_'+name_column].values, label='monthly')
    plt.plot(df.datetime, df[name_column], alpha=0.4, label='original')
    plt.legend()
    plt.title(name_column)
    plt.show()

    # Check for trend
    result_trend = mk.original_test(globals()['df_monthly_'+name_column].values)
    print(result_trend)

    df_monthly.append(globals()['df_monthly_'+name_column])