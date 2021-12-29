url = f"{sdc_}start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"
url = f"{url}&instrument_id={var['inst']}&data_rate_mode={var['tmmode']}"
url = f"&data_level={var['lev']}"

if user is None:
    url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/file_info/science?start_date=' + start_date + '&end_date=' + end_date + '&sc_id=mms' + prb + '&instrument_id=' + instrument + '&data_rate_mode=' + drate + '&data_level=' + lvl
else:
url = f"{sdc_}start_date=" + start_date + '&end_date=' + end_date + \
      '&sc_id=mms' + prb + '&instrument_id=' + instrument + '&data_rate_mode=' + drate + '&data_level=' + lvl

if dtype != '':
    url = url + '&descriptor=' + dtype

if CONFIG['debug_mode']: logging.info('Fetching: ' + url)

if no_download == False:
    # query list of available files
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            http_json = sdc_session.get(url, verify=True,
                                        headers=headers).json()