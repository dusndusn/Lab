import pandas as pd
import numpy as np
from tqdm import tqdm

path = r'Z:/03-Lab-meeting/2024-Tutorial/physionet_org/files/mimiciii/1_4'

#ICUSTAY: 
#remove ICU stays whose duration is either less than 24 hours or more than 48 hours
icustays=pd.read_csv(f'{path}/ICUSTAYS.csv', usecols=['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS'])
filtered_icustays=icustays[(icustays['LOS']>=1)&(icustays['LOS']<=2)]

#load admission data
admissions=pd.read_csv(f'{path}/ADMISSIONS.csv', usecols=['HADM_ID','DEATHTIME'])
admissions.loc[:,'DEATHTIME']=pd.to_datetime(admissions['DEATHTIME'])


filtered_icustays.loc[:,'INTIME']=pd.to_datetime(filtered_icustays['INTIME'])
filtered_icustays.loc[:,'OUTTIME']=pd.to_datetime(filtered_icustays['OUTTIME'])

#merge deathtime to filtered_icustays
filtered_icustays=filtered_icustays.merge(admissions[['HADM_ID', 'DEATHTIME']], on='HADM_ID', how='left')

def is_dead(row):
    if pd.isna(row['DEATHTIME']):
        return 0
    return int(row['INTIME']<=row['DEATHTIME']<=row['OUTTIME'])

filtered_icustays['DEATH']=filtered_icustays.apply(is_dead, axis=1)
print(filtered_icustays.head())


#CHARTEVENTS: 
#disregard NULL icustay_id and valuenum
filtered_chartevents=pd.DataFrame()
chuncksize=10**6

for chunck in tqdm(pd.read_csv(f'{path}/CHARTEVENTS.csv', usecols=['ICUSTAY_ID','CHARTTIME','ITEMID','VALUENUM'], chunksize=chuncksize)):
    filtered_chunck=chunck.dropna(subset=['ICUSTAY_ID', 'VALUENUM'])
    filtered_chartevents=pd.concat([filtered_chartevents, filtered_chunck], ignore_index=True)

#sort by time then consider the first 3 hours, 100 at max each
filtered_chartevents = filtered_chartevents[filtered_chartevents['ICUSTAY_ID'].isin(filtered_icustays['ICUSTAY_ID'])]
filtered_chartevents['CHARTTIME']=pd.to_datetime(filtered_chartevents['CHARTTIME'])
filtered_chartevents=filtered_chartevents.sort_values(by=['ICUSTAY_ID','CHARTTIME'])
filtered_chartevents['TIME_DIF']=filtered_chartevents.groupby('ICUSTAY_ID')['CHARTTIME'].transform(lambda x: (x-x.min()).dt.total_seconds()/3600)
filtered_chartevents=filtered_chartevents[filtered_chartevents['TIME_DIF']<=3]

#print(filtered_chartevents.head(150))
icustay_counts=filtered_chartevents.groupby('ICUSTAY_ID').size()
print(icustay_counts.reset_index(name='counts'))

'''
filtered_chartevents=filtered_chartevents.groupby('ICUSTAY_ID').head(100)

#one_hot_encoding
top_itemids=filtered_chartevents['ITEMID'].value_counts().nlargest(100).index
filtered_chartevents=filtered_chartevents[filtered_chartevents['ITEMID'].isin(top_itemids)]

def encode_itemid(df):
    pivot_df=df.pivot_table(index='ICUSTAY_ID', columns='ITEMID', values='VALUENUM', aggfunc='mean', fill_value=0)
    return pivot_df.astype(np.float32)

encoded_chartevents=encode_itemid(filtered_chartevents)
encoded_chartevents=encoded_chartevents.reset_index()

#filtered_chartevents['ICUSTAY_ID']=filtered_chartevents['ICUSTAY_ID'].astype(int)
#filtered_chartevents['last_digit'] = filtered_chartevents['ICUSTAY_ID'].astype(str).str[-1].astype(int)

#encoded_chartevents=encoded_chartevents.merge(filtered_chartevents[['ICUSTAY_ID', 'last_digit']].drop_duplicates(), on='ICUSTAY_ID', how='left')
encoded_chartevents=encoded_chartevents.merge(filtered_icustays[['ICUSTAY_ID', 'DEATH']].drop_duplicates(),on='ICUSTAY_ID', how='inner')

encoded_chartevents['ICUSTAY_ID']=encoded_chartevents['ICUSTAY_ID'].astype(int)
encoded_chartevents['last_digit']=encoded_chartevents['ICUSTAY_ID'].astype(str).str[-1].astype(int)



train_encoded=encoded_chartevents[~encoded_chartevents['last_digit'].isin([8,9])]
test_encoded=encoded_chartevents[encoded_chartevents['last_digit'].isin([8,9])]


x_train=train_encoded.drop(columns=['ICUSTAY_ID', 'last_digit', 'DEATH']).to_numpy()
x_test=test_encoded.drop(columns=['ICUSTAY_ID','last_digit', 'DEATH']).to_numpy()

#y_train=encoded_chartevents[~encoded_chartevents['ICUSTAY_ID'].astype(str).str[-1].isin(['8','9'])]['DEATH']
#y_test=encoded_chartevents[encoded_chartevents['ICUSTAY_ID'].astype(str).str[-1].isin(['8','9'])]['DEATH']

y_train=train_encoded['DEATH']
y_test=test_encoded['DEATH']

#X_train_logistic.npy and x_test_logistic.npy

np.save('x_train_logistic.npy', x_train)
np.save('x_test_logistic.npy', x_test)



#x_train_rnn.npy, x_test_rnn.npy
def rnn_format(df):
    format=df.groupby('ICUSTAY_ID').apply(
        lambda group: ' '.join(
            f"{time:.1f}:{itemid}:{value}"
            for time, itemid, value in zip(group['TIME_DIF']*60, group['ITEMID'], group['VALUENUM'])
        )
    )
    return format

x_train_rnn=rnn_format(filtered_chartevents[filtered_chartevents['ICUSTAY_ID'].isin(train_encoded['ICUSTAY_ID'])])
x_test_rnn=rnn_format(filtered_chartevents[filtered_chartevents['ICUSTAY_ID'].isin(train_encoded['ICUSTAY_ID'])])

np.save('x_train_rnn.npy', np.array(x_train_rnn))
np.save('x_test_rnn.npy', np.array(x_test_rnn))


np.save('y_train.npy', y_train.to_numpy(dtype=np.float32))
np.save('y_test.npy', y_test.to_numpy(dtype=np.float32))

'''
