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


filtered_chartevents=filtered_chartevents.merge(filtered_icustays[['ICUSTAY_ID','DEATH']], on='ICUSTAY_ID', how='inner')
filtered_chartevents['ICUSTAY_ID']=filtered_chartevents['ICUSTAY_ID'].astype(int)
filtered_chartevents['last_digit']=filtered_chartevents['ICUSTAY_ID'].astype(str).str[-1].astype(int)


#modify ITEM_ID:
#choose ITEM_ID
select_data=filtered_chartevents[~filtered_chartevents['last_digit'].isin([8,9])]
count_id=select_data.groupby('ITEMID')['ICUSTAY_ID'].nunique().reset_index(name='count')
total_icustays=select_data['ICUSTAY_ID'].nunique()
count_id=count_id[count_id['count']>=0.4*total_icustays]
filtered_chartevents=filtered_chartevents[filtered_chartevents['ITEMID'].isin(count_id['ITEMID'])]
filtered_chartevents['TIME_BUCKET']=filtered_chartevents['TIME_DIF']*3600//108

def normalization(group):
    min_val=group['VALUENUM'].min()
    max_val=group['VALUENUM'].max()
    if min_val!=max_val:
        group['VALUENUM']=(group['VALUENUM']-min_val)/(max_val-min_val)
    else:
        group['VALUENUM']=np.nan
    return group

filtered_chartevents=filtered_chartevents.groupby('ITEMID', group_keys=False).apply(normalization)
filtered_chartevents=filtered_chartevents.sort_values(by=['ICUSTAY_ID', 'TIME_BUCKET'])

#one_hot_encoding
encoded_chartevent=pd.get_dummies(filtered_chartevents['ITEMID'], prefix='ITEMID')

encoded_chartevent=encoded_chartevent.mul(filtered_chartevents['VALUENUM'], axis=0)
filtered_chartevents=pd.concat([filtered_chartevents, encoded_chartevent], axis=1)
filtered_chartevents=filtered_chartevents.drop(columns=['ITEMID'])

#data_RNN_model:
#10rows for each ICUSTAY_ID
aggregated = filtered_chartevents.groupby(['ICUSTAY_ID', 'TIME_BUCKET'], as_index=False).mean()
pivoted = aggregated.pivot_table(index=['ICUSTAY_ID', 'TIME_BUCKET'], columns='ITEMID', values='VALUENUM', fill_value=0)
all_buckets = np.arange(100)
all_itemids = aggregated.columns[4:]
idx = pd.MultiIndex.from_product([pivoted.index.levels[0], all_buckets], names=['ICUSTAY_ID', 'TIME_BUCKET'])
pivoted = pivoted.reindex(idx, fill_value=0).reset_index()
pivoted = pivoted.fillna(0).sort_index(axis=1)
death_df = filtered_icustays[['ICUSTAY_ID', 'DEATH']].drop_duplicates()
rnn_chartevents = death_df.merge(pivoted, on='ICUSTAY_ID', how='left')
rnn_chartevents=rnn_chartevents.merge(filtered_chartevents[['last_digit']], on='ICUSTAY_ID', how='left')
print(rnn_chartevents.head(100))

#divide in train set and test set
train_encoded=rnn_chartevents[~rnn_chartevents['last_digit'].isin([8,9])]
test_encoded=rnn_chartevents[rnn_chartevents['last_digit'].isin([8,9])]

train_encoded=train_encoded.drop(columns=['ICUSTAY_ID', 'last_digit', 'DEATH'])
test_encoded=test_encoded.drop(columns=['ICUSTAY_ID', 'last_digit', 'DEATH'])

np.save('x_train_rnn.npy', np.array(train_encoded))
np.save('x_test_rnn.npy', np.array(test_encoded))

#data_logistic_regression_model:
#combine rows for the same ICUSTAY_ID
#divide in train set and test set
#y_train, y_test using death column
