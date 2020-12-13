import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Acessando o dataset padrão
file = "datasetLoL\\matchinfo.csv"
dataset = pd.read_csv(file)

#Problema 1: Criar um dataset com todos os jogos de todas as ligas com suas correspondentes composições
matchup=pd.DataFrame(columns=['Liga','Ano','Split','Time','Top','Jungle','Mid','ADC','Suporte'])
matchup['Liga']=np.concatenate((dataset.League,dataset.League))
matchup['Ano']=np.concatenate((dataset.Year,dataset.Year))
matchup['Split']=np.concatenate((dataset.Season,dataset.Season))
matchup['Time']=np.concatenate((dataset.redTeamTag,dataset.blueTeamTag))
matchup['Top']=np.concatenate((dataset.redTopChamp,dataset.blueTopChamp))
matchup['Jungle']=np.concatenate((dataset.redJungleChamp,dataset.blueJungleChamp))
matchup['Mid']=np.concatenate((dataset.redMiddleChamp,dataset.blueMiddleChamp))
matchup['ADC']=np.concatenate((dataset.redADCChamp,dataset.blueADCChamp))
matchup['Suporte']=np.concatenate((dataset.redSupportChamp,dataset.blueSupportChamp))
matchup.head(10)

#Problema 2: Match-up com os Campeões mais utilizados pela PAIN Gaming
print("Campeões mais Utilizados pela PAIN Gaming:")
print("Top:",(matchup[(matchup.Liga == 'CBLoL') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'PNG')].Top.describe().top),"||",
      "Jungle:",(matchup[(matchup.Liga == 'CBLoL') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'PNG')].Jungle.describe().top),"||",
      "Mid:",(matchup[(matchup.Liga == 'CBLoL') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'PNG')].Mid.describe().top),"||",
      "ADC:",(matchup[(matchup.Liga == 'CBLoL') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'PNG')].ADC.describe().top),"||",
      "Suporte:",(matchup[(matchup.Liga == 'CBLoL') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'PNG')].Suporte.describe().top))

#Problema 3: Match-up com os Campeões mais utilizados pela SKT T1
print("\nCampeões mais Utilizados pela SKT T1:")
print("Top:",(matchup[(matchup.Liga == 'LCK') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'SKT')].Top.describe().top),"||",
      "Jungle:",(matchup[(matchup.Liga == 'LCK') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'SKT')].Jungle.describe().top),"||",
      "Mid:",(matchup[(matchup.Liga == 'LCK') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'SKT')].Mid.describe().top),"||",
      "ADC:",(matchup[(matchup.Liga == 'LCK') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'SKT')].ADC.describe().top),"||",
      "Suporte:",(matchup[(matchup.Liga == 'LCK') & (matchup.Ano == 2017) & (matchup.Split == 'Summer') & (matchup.Time == 'SKT')].Suporte.describe().top))

#Problema 4: Match-up com os Campeões mais utilizados em todo o Competitivo de LoL (2015 a 2018)
champions = [matchup.Top.describe().top,matchup.Jungle.describe().top,matchup.Mid.describe().top,matchup.ADC.describe().top,matchup.Suporte.describe().top]
picks = [matchup.Top.describe().freq,matchup.Jungle.describe().freq,matchup.Mid.describe().freq,matchup.ADC.describe().freq,matchup.Suporte.describe().freq]
plt.bar(champions,picks,color="red")
plt.title("Campeões mais Utilizados")
plt.xlabel("Campeões por Lane (Top/Jungle/Mid/ADC/Sup)")
plt.ylabel("Quantidade de Picks")
plt.show()

#Problema 5: Tempo médio de partida por liga
dataset.groupby('League').gamelength.mean().plot(kind='bar')
plt.title('Tempo médio de partida por liga')
plt.xlabel('Ligas')
plt.ylabel('Tempo médio em minutos')
plt.show()

#Problema 6: WinRate (percentual de vitórias) do lado Vermelho e lado Azul referente ao Competitivo de 2015 a 2018
pazul = '{:2.2}'.format(dataset.bResult.mean())
pvermelho = '{:2.2}'.format(dataset.rResult.mean())
plt.pie([pazul,pvermelho],labels=['Lado Azul', 'Lado Vermelho'],colors=['blue', 'red'],autopct='%1.1f%%')
plt.title('WinRate dos Lados')
plt.show()

#Problema 7: Quantidade de campeões diferentes utilizados em cada lane
top=matchup.Top.describe()
jg=matchup.Jungle.describe()
mid=matchup.Mid.describe()
adc=matchup.ADC.describe()
sup=matchup.Suporte.describe()
plt.bar(['Top','Jungle','Mid','ADC','Suporte'], [top['unique'],jg['unique'],mid['unique'],adc['unique'],sup['unique']], color='red')
plt.xlabel('Lanes')
plt.ylabel('Quantidade de Campeões')
plt.title('Variedade de Campeões por Lane')
plt.show()