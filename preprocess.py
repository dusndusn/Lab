import pandas as pd
import numpy as np

path = r'Z:/03-Lab-meeting/2024-Tutorial/physionet_org/files/mimiciii/1_4'

#ICUSTAY: remove ICU stays whose duration is either less than 24 hours or more than 48 hours
icustays=pd.read_csv(f'{path}/ICUSTAYS.csv')
filtered_icustays=icustays[(icustays['LOS']>=1)&(icustays['LOS']<=2)]

#CHARTEVENTS: disregard NULL icustay_id and valuenum
filtered_chartevents=pd.DataFrame()
chuncksize=10**6

for chunck in pd.read_csv(f'{path}/CHARTEVENTS.csv', usecols=['ICUSTAY_ID','CHARTTIME','ITEMID','VALUENUM'], chunksize=chuncksize):
    filtered_chunck=chunck.dropna(subset=['ICUSTAY_ID', 'VALUENUM'])
    filtered_chartevents=pd.concat([filtered_chartevents, filtered_chunck], ignore_index=True)

filtered_chartevents['CHARTTIME']=pd.to_datetime(filtered_chartevents['CHARTTIME'])

#CHARTEVENTS: sort by time then consider the first 3 hours, 100 at max each
filtered_chartevents = filtered_chartevents[filtered_chartevents['ICUSTAY_ID'].isin(filtered_icustays['ICUSTAY_ID'])]
filtered_chartevents=filtered_chartevents.sort_values(by=['ICUSTAY_ID','CHARTTIME'])

filtered_chartevents['TIME_DIF']=filtered_chartevents.groupby('ICUSTAY_ID')['CHARTTIME'].transform(lambda x: (x-x.min()).dt.total_seconds()/3600)
filtered_chartevents=filtered_chartevents[filtered_chartevents['TIME_DIF']<=3]
filtered_chartevents=filtered_chartevents.groupby('ICUSTAY_ID').head(100)
filtered_charevents=filtered_chartevents.drop_duplicates(subset=['ICUSTAY_ID', 'TIME_DIF', 'ITEMID'], keep='last')

#encoding

#X_train_logistic.npy and x_test_logistic.npy
filtered_chartevents['last_digit']=filtered_charevents['ICUSTAY_ID'].astype(str).str[-1].astype(int)
train_chartevents=filtered_chartevents[~filtered_chartevents['last_digit'].isin([8,9])]
test_chartevents=filtered_chartevents[filtered_chartevents['last_digit'].isin([8,9])]

x_train_logistic=train_chartevents.pivot_table(index='ICUSTAY_ID', columns='ITEMID', values='VALUENUM', aggfunc='last').fillna(0)
x_test_logistic=test_chartevents.pivot_table(index='ICUSTAY_ID', columns='ITEMID', values='VALUENUM', aggfunc='last').fillna(0)

np.save('x_train_logistic.npy', x_train_logistic.to_numpy())
np.save('x_test_logistic.npy', x_test_logistic.to_numpy())


#x_train_rnn.npy, x_test_rnn.npy
def rnn_format(df):
    format=df.groupby('ICUSTAY_ID').apply(
        lambda group: ' '.join(
            f"{time:.1f}:{itemid}:{value}"
            for time, itemid, value in zip(group['TIME_DIF']*60, group['ITEMID'], group['VALUENUM'])
        )
    )
    return format

x_train_rnn=rnn_format(train_chartevents)
x_test_rnn=rnn_format(test_chartevents)

np.save('x_train_rnn.npy', x_train_rnn)
np.save('x_test_rnn.npy', x_test_rnn)

#load admission data
admissions=pd.read_csv(f'{path}/ADMISSIONS.csv', usecols=['HADM_ID','DEATHTIME'])
admissions['DEATHTIME']=pd.to_datetime(admissions['DEATHTIME'])

filtered_icustays=filtered_icustays.merge(admissions[['HADM_ID', 'DEATHTIME']], on='HADM_ID', how='left')
filtered_icustays['INTIME']=pd.to_datetime(filtered_icustays['INTIME'])
filtered_icustays['OUTTIME']=pd.to_datetime(filtered_icustays['OUTTIME'])


def is_dead(row):
    if pd.isna(row['DEATHTIME']):
        return 0
    return int(row['INTIME']<=row['DEATHTIME']<=row['OUTTIME'])

filtered_icustays['DEATH']=filtered_icustays.apply(is_dead, axis=1)


#y_train.npy, y_test.npy
y_train=filtered_icustays[~filtered_icustays['ICUSTAY_ID'].astype(str).str[-1].isin(['8','9'])]['DEATH']
y_test=filtered_icustays[filtered_icustays['ICUSTAY_ID'].astype(str).str[-1].isin(['8','9'])]['DEATH']

np.save('y_train.npy', y_train.to_numpy(dtype=np.float32))
np.save('y_test.npy', y_test.to_numpy(dtype=np.float32))
