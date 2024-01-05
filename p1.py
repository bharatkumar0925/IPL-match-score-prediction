import pandas as pd
import numpy as np

path = r"C:\Users\BHARAT\Desktop\data sets\ipl-all-season\match_data.csv"
path1 = r"C:\Users\BHARAT\Desktop\data sets\ipl-all-season\match_info_data.csv"
data = pd.read_csv(path, usecols=list(range(0, 13))+[18])
summary = pd.read_csv(path1, usecols=range(0, 10))
data = data.drop(['start_date', 'striker', 'non_striker', 'bowler'], axis=1)
data = data.rename(columns={'runs_off_bat': 'runs', 'extras': 'extra', 'match_id': 'id'})
summary['season'] = summary['season'].replace({'2009/10': 2010, '2020/21': 2020, '2007/08': 2008})
data = data.replace({
    'Rising Pune Supergiants': 'Chennai Super Kings',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Chennai Super Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Rajasthan Royals',
    'Wankhede Stadium': 'Wankhede Stadium, Mumbai',
    'Eden Gardens': 'Eden Gardens, Kolkata',
    'M Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
    'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium, Delhi',
    'Arun Jaitley Stadium': 'Arun Jaitley Stadium, Delhi',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
    'Sawai Mansingh Stadium': 'Sawai Mansingh Stadium, Jaipur',
    'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
    'Himachal Pradesh Cricket Association Stadium': 'Himachal Pradesh Cricket Association Stadium, Dharamsala',
'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
    'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium,',
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
    'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium, Pune',
    'Brabourne Stadium': 'Brabourne Stadium, Mumbai',

})
data['is_wicket'] = data['wicket_type'].notnull().astype('int8')
data['wickets'] = data.groupby(['id', 'innings'])['is_wicket'].cumsum()
data['wickets_left'] = 10-data['wickets']


group = data.groupby(['id', 'innings'])[['runs', 'extra']].sum().reset_index()
group['total'] = group['runs']+group['extra']
current = data.groupby(['id', 'innings'])[['runs', 'extra']].cumsum()
data['current_score'] = current['runs']+current['extra']

group.drop(['runs', 'extra'], axis=1, inplace=True)
current.drop(['runs', 'extra'], axis=1, inplace=True)

data = data.merge(group, on=['id', 'innings'])

data[['n1', 'n2']] = data['ball'].astype('str').str.split('.', expand=True).astype('int')
data['ball'] = data['n1']*6+data['n2']
data['balls_left'] = 120-data['ball']
data['crr'] = (data['current_score']*6/data['ball']).round(2)

print(data[['ball', 'current_score', 'crr']])





data.drop(['season', 'wicket_type', 'extra', 'n1', 'n2'], axis=1, inplace=True)
data = summary.merge(data, on='id')
teams = ['Kochi Tuskers Kerala', 'Pune Warriors']
data = data.query(' innings==1 and dl_applied != 1 and (team1 != @teams and team2 != @teams)')
data = data.replace({'Bangalore': 'Bengaluru'})
data['season'] = data['season'].astype('int')
data['avg_runs_city'] = data.groupby('city')['total'].transform('median').round(2)
data['avg_runs_city'] = data['avg_runs_city'].ffill()

print(data['total'].describe())
data['last5_runs'] = data.groupby('id')['runs'].transform(lambda x: x.rolling(window=30).sum())
#data['last5_wicketts'] = data.groupby('id')['is_wicket'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
#data['last3_runs'] = data.groupby('id')['runs'].transform(lambda x: x.rolling(window=18, min_periods=1).sum())
#data['last3_wickets'] = data.groupby('id')['is_wicket'].transform(lambda x: x.rolling(window=18, min_periods=1).sum())
#data['last5_overs'] = np.where(data['ball'] > 90, True, False)
data['last5_runs'] = data['last5_runs'].ffill()
data.dropna(inplace=True)
#print(data[['last5_runs', 'last5_wicketts']].head(40))

#print(data.info())
data.drop(['is_wicket', 'runs', 'innings'], axis=1, inplace=True)
print(data.isnull().sum())
data.to_csv('matches.csv', index=False)


d = data.groupby('batting_team').agg({'last5_runs': ['idxmin', 'min']})
print(d)

da = data.drop_duplicates('city').reset_index()
print(da[['city', 'avg_runs_city']])

