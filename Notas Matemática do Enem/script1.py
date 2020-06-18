""" 
        Criar modelo de regressão capaz de fazer a predição das notas de MAT
     do enem 2016. É um projeto importante, Victor! Foco!

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


treino_enem = pd.read_csv('train.csv')
test_enem = pd.read_csv('test.csv')

# =============================================================================
#     Para análise exploratória, optou-se por começar com as correlações,
# já que o volume de dados é muito grande, então será mais fácil de filtrar
# =============================================================================

correlacao = treino_enem.corr()

# =============================================================================
#     Agora faremos uma lista com os atributos cujas correlações com a nota de
# matemática é maior que 0.3 (o ideal seria no mínimo 0.5, mas assim o volume 
# de dados reduziria muito)
# =============================================================================

features = correlacao[(correlacao['NU_NOTA_MT'] <= -0.3) | (correlacao['NU_NOTA_MT'] >= 0.3) 
                       & (correlacao['NU_NOTA_MT'] < 1.0)]['NU_NOTA_MT']

features = features.drop(['Q038', 'Q037'])
features_list = features.index.to_list()

# =============================================================================
#     Filtrando os nulls
# =============================================================================

# preenchendo com 0
df1_treino = treino_enem.copy()
df1_teste = test_enem.copy()

df1_treino = df1_treino.fillna(0)
df1_teste = df1_teste.fillna(0)

#preenchendo com médias
df2_treino = treino_enem.copy()
df2_teste = test_enem.copy()

df2_treino = df2_treino.fillna(df2_treino.mean())
df2_teste = df2_teste.fillna(df2_teste.mean())

#preenchendo com medianas
df3_treino = treino_enem.copy()
df3_teste = test_enem.copy()

df3_treino = df3_treino.fillna(df3_treino.median())
df3_teste = df3_teste.fillna(df3_teste.median())

#preenchendo com -1
df4_treino = treino_enem.copy()
df4_teste = test_enem.copy()

df4_treino = df4_treino.fillna(-1)
df4_teste = df4_teste.fillna(-1)

# =============================================================================
#     Configurando pipelines para seleção do algoritmo
# =============================================================================

x_treino = df1_treino[features_list]
y_treino = df1_treino['NU_NOTA_MT']
x_teste = df1_teste[features_list]

pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', MinMaxScaler()),('LR',LinearRegression())])))
pipelines.append(('KNN', Pipeline([('Scaler', MinMaxScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('DTR', Pipeline([('Scaler', MinMaxScaler()),('DTR', DecisionTreeRegressor())])))
pipelines.append(('GBM', Pipeline([('Scaler', MinMaxScaler()),('GBM', GradientBoostingRegressor())])))
pipelines.append(('RFR', Pipeline([('Scaler', MinMaxScaler()),('RFR', RandomForestRegressor())])))

# =============================================================================
#     Análise de métricas para seleção algoritmo
# =============================================================================

def validaPerformanceModelos(pipelines,x_treino,y_treino):
    results = []
    names = []
    for name, model in pipelines:
        kfold = RepeatedKFold(n_splits=4, n_repeats= 15, random_state = 0)
        cv_results = cross_val_score(model, x_treino, y_treino, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
validaPerformanceModelos(pipelines, x_treino, y_treino)
#os modelos escolhidos foram GradientBoostinRegressor e RandomForestRegressor

# =============================================================================
#     GradientBoostRegressor
# =============================================================================

pipe_GBM = Pipeline([('scaler',  StandardScaler()),
            ('GradientBoostingRegressor', GradientBoostingRegressor())])
CV_pipe_GBM = GridSearchCV(estimator = pipe_GBM, param_grid = {},cv = 5,return_train_score=True, verbose=0)

CV_pipe_GBM.fit(x_treino, y_treino)
p = CV_pipe_GBM.predict(x_teste)

# =============================================================================
#     Random Forest Regressor
# =============================================================================

pipe_RFR = Pipeline([('scaler',  StandardScaler()),
            ('RandomForestRegressor', RandomForestRegressor())])

CV_pipe_RFR = GridSearchCV(estimator = pipe_RFR, param_grid = {},cv = 5,return_train_score=True, verbose=0)


CV_pipe_RFR.fit(x_treino, y_treino)
p = CV_pipe_RFR.predict(x_teste)

# =============================================================================
#     Criando dataset de respostas
# =============================================================================

df_result = pd.DataFrame()
df_result['NU_INSCRICAO'] = test_enem['NU_INSCRICAO']
df_result['NU_NOTA_MT'] = np.around(p, 2)
df_result.to_csv('answer.csv', index= False, header= True)