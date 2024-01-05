import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import warnings
import pickle
warnings.filterwarnings('ignore')


data = pd.read_csv('matches.csv')
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday

data.drop(['ball', 'balls_left', 'result', 'dl_applied', 'date'], axis=1, inplace=True)
#data = data.query('season != 2018')

cat_col = list(data.select_dtypes(['object', 'category']))
data = data.astype({'id': 'int32', **{col: 'category' for col in cat_col}})

# Apply OneHotEncoder using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, dtype='bool', drop='if_binary'), cat_col)
    ],
    remainder='passthrough'
)

data = data.astype({'current_score': 'int16', 'id': 'int32', 'crr': 'float32', 'total': 'int16', 'wickets': 'int8', 'wickets_left': 'int8', 'season': 'int16'})

X = data.drop('total', axis=1)
y = data['total']
our_data = data.query('id==1304047')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base = [
('rf', RandomForestRegressor(n_estimators=100, max_samples=0.8, max_features=5, n_jobs=-1, oob_score=True, random_state=42)),
('rr', Ridge(max_iter=1000, random_state=42, positive=True, solver='lbfgs', alpha=15)),
('knn', KNeighborsRegressor(weights='distance', n_jobs=-1, metric='euclidean')),
    ('xgb', xgb.XGBRegressor(n_estimators=100, n_jobs=-1, max_depth=50)),
    ('gbr', GradientBoostingRegressor(n_estimators=25, learning_rate=0.3, max_depth=70, loss='quantile', random_state=42)),
    ('ada', AdaBoostRegressor())
]
final = GradientBoostingRegressor(max_depth=10d)

model = StackingRegressor(base, final, cv=3, n_jobs=-1)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
#cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
pipe.fit(X_train, y_train)

#oob_score = pipe.named_steps['model'].oob_score_

with open('score_prediction.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)

y_pred = pipe.predict(X_test).astype(int)
#y_pred = np.where(X_test['wickets_left'] == 0,
#                  X_test['current_score'],
#                  pipe.predict(X_test).astype(int))

print(r2_score(y_pred, y_test)*100)
print(mean_squared_error(y_pred, y_test))
#print(oob_score*100)
#print(round(cv.mean()*(100), 3))


df = our_data.drop('total', axis=1)
prediction = pipe.predict(df).round(0).astype(int)

our_data['predicted'] = prediction
print(our_data[['current_score', 'wickets', 'total', 'predicted']].to_string(index=False))

