import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../datasetLoL/LeagueofLegends.csv')
df.head()
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval


#Acessando o dataset padrão
lol = "..\\datasetLoL\\LeagueofLegends.csv"
dataset = pd.read_csv(lol)
print (dataset['goldblue'].apply(max))
#print(dataset.info()) #Analisando as colunas do dataset e verificando os tipos

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis') #Verificando a presença de dados nulos
#plt.show()

dsAux = dataset.copy(deep=True)

#Colunas a que influenciam na vitórias de uma partida: gold, kills, towers, inhibs, dragons, barons, heralds

dsAux['goldred'] = dsAux['goldred'].apply(literal_eval)
dsAux['goldblue'] = dsAux['goldblue'].apply(literal_eval)
dsAux['rKills'] = dsAux['rKills'].apply(literal_eval)
dsAux['bKills'] = dsAux['bKills'].apply(literal_eval)
dsAux['rTowers'] = dsAux['rTowers'].apply(literal_eval)
dsAux['bTowers'] = dsAux['bTowers'].apply(literal_eval)
dsAux['rInhibs'] = dsAux['rInhibs'].apply(literal_eval)
dsAux['bInhibs'] = dsAux['bInhibs'].apply(literal_eval)
dsAux['rDragons'] = dsAux['rDragons'].apply(literal_eval)
dsAux['bDragons'] = dsAux['bDragons'].apply(literal_eval)
dsAux['rBarons'] = dsAux['rBarons'].apply(literal_eval)
dsAux['bBarons'] = dsAux['bBarons'].apply(literal_eval)
dsAux['rHeralds'] = dsAux['rHeralds'].apply(literal_eval)
dsAux['bHeralds'] = dsAux['bHeralds'].apply(literal_eval)

partidas = pd.DataFrame()

partidas['tagAzul'] = dsAux['blueTeamTag']
partidas['resultAzul'] = dsAux['bResult']
partidas['goldAzul'] = dsAux['goldblue'].apply(max)
partidas['killsAzul'] = dsAux['bKills'].apply(len)
partidas['towersAzul'] = dsAux['bTowers'].apply(len)
partidas['inibsAzul'] = dsAux['bInhibs'].apply(len)
partidas['dragonsAzul'] = dsAux['bDragons'].apply(len)
partidas['baronsAzul'] = dsAux['bBarons'].apply(len)
partidas['heraldsAzul'] = dsAux['bHeralds'].apply(len)

partidas['tagVerm'] = dsAux['redTeamTag']
partidas['resultVerm'] = dsAux['rResult']
partidas['goldVerm'] = dsAux['goldred'].apply(max)
partidas['killsVerm'] = dsAux['rKills'].apply(len)
partidas['towersVerm'] = dsAux['rTowers'].apply(len)
partidas['inibsVerm'] = dsAux['rInhibs'].apply(len)
partidas['dragonsVerm'] = dsAux['rDragons'].apply(len)
partidas['baronsVerm'] = dsAux['rBarons'].apply(len)
partidas['heraldsVerm'] = dsAux['rHeralds'].apply(len)

partidas = partidas[(partidas.tagAzul == 'PNG') | (partidas.tagVerm == 'PNG')]
partidas = partidas.reset_index(drop=True)
partidas['timeVencedor'] = np.where(partidas.resultAzul == 1, 1, 2)

print(partidas.head())

#Divisão de treino-teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(partidas[['goldAzul','killsAzul','towersAzul','inibsAzul','dragonsAzul','baronsAzul','heraldsAzul',
                                                              'goldVerm','killsVerm','towersVerm','inibsVerm','dragonsVerm','baronsVerm','heraldsVerm']],
                                                    partidas['timeVencedor'],test_size=0.3,random_state=42)
#Treinando o modelo

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

ax = sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d',xticklabels=['Red Team','Blue Team'], yticklabels=['Red Team','Blue Team'])
ax.set(xlabel = 'True Label', ylabel='Predicted')
plt.show()





