import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Acessando o dataset padrão
file = "datasetLoL\\matchinfo.csv"
dataset = pd.read_csv(file)

#Criando o dataset com informações referentes ao CBLol 2017 Summer
CBLol2017 = dataset[(dataset.League == 'CBLoL') & (dataset.Year == 2017) & (dataset.Season == 'Summer')]

#Problema 1: Listar todos os jogos da liga desejada com suas correspondentes composições
matchup=pd.DataFrame(columns=['Time','Top','Jungle','Mid','ADC','Suporte'])
matchup['Top']=np.concatenate((CBLol2017.redTopChamp,CBLol2017.blueTopChamp));
matchup['Jungle']=np.concatenate((CBLol2017.redJungleChamp,CBLol2017.blueJungleChamp));
matchup['Mid']=np.concatenate((CBLol2017.redMiddleChamp,CBLol2017.blueMiddleChamp));
matchup['ADC']=np.concatenate((CBLol2017.redADCChamp,CBLol2017.blueADCChamp));
matchup['Suporte']=np.concatenate((CBLol2017.redSupportChamp,CBLol2017.blueSupportChamp));
matchup['Time']=np.concatenate((CBLol2017.redTeamTag,CBLol2017.blueTeamTag));
print(matchup.head(20))

#Problema 2: Match-up com os Campeões mais utilizados pela PAIN Gaming
print("Top:",(matchup[(matchup.Time == 'PNG')].Top.describe().top),"||","Jungle:",(matchup[(matchup.Time == 'PNG')].Jungle.describe().top),"||","Mid:",(matchup[(matchup.Time == 'PNG')].Mid.describe().top),"||","ADC:",(matchup[(matchup.Time == 'PNG')].ADC.describe().top),"||","Suporte:",(matchup[(matchup.Time == 'PNG')].Suporte.describe().top))

#Problema 3: Match-up com os Campeões mais utilizados no CBLol 2017 Summer
print("Top:",(matchup.Top.describe().top),"||","Jungle:",(matchup.Jungle.describe().top),"||","Mid:",(matchup.Mid.describe().top),"||","ADC:",(matchup.ADC.describe().top),"||","Suporte:",(matchup.Suporte.describe().top))

#Problema 4: Tempo médio de partida por liga
dataset.groupby('League').gamelength.mean().plot(kind='bar')
plt.title('Tempo médio de partida por liga')
plt.show()

#Problema 5: WinRate (percentual de vitórias) do lado Vermelho e lado Azul referente ao Competitivo de 2015 a 2018
pazul = '{:2.2}'.format(dataset.bResult.mean())
pvermelho = '{:2.2}'.format(dataset.rResult.mean())
plt.pie([pazul,pvermelho],labels=['Lado Azul', 'Lado Vermelho'],colors=['blue', 'red'],autopct='%1.1f%%')
plt.title('WinRate dos Lados')
plt.show()
