import pandas as pd
import numpy as np
import pickle

from Utils import melt_df, plota_indices,apply_custom_css,evaluate_model,plot_corr,prever_evasao

import streamlit as st



# Extract

path_csv = './data/Cópia de PEDE_PASSOS_DATASET_FIAP.csv'
dataset = pd.read_csv(path_csv, sep=";")

# Transform

# Feature engineering

index_to_drop = dataset[dataset['IDADE_ALUNO_2020'] == 'D108'].index
dataset.drop(index_to_drop, inplace=True)

dataset[['INDE_2021']] = dataset[['INDE_2021']].replace('#NULO!',np.nan)
colunas_para_converter = [ 'IDADE_ALUNO_2020', 'ANOS_PM_2020', 'INDE_2020', 'IAA_2020', 'IEG_2020', 
                           'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020', 'INDE_2021' ]
dataset[colunas_para_converter] = dataset[colunas_para_converter].astype('float64')

dataset['FASE_2020'] = dataset['FASE_TURMA_2020'].str[0]
dataset['TURMA_2020'] = dataset['FASE_TURMA_2020'].str[1:]
dataset['FASE_2020'] = dataset['FASE_2020'].astype(float)

df = dataset[['NOME', 'IDADE_ALUNO_2020','ANOS_PM_2020','SINALIZADOR_INGRESSANTE_2021','ANO_INGRESSO_2022',
       'FASE_2020', 'TURMA_2020', 'INSTITUICAO_ENSINO_ALUNO_2020', 'PONTO_VIRADA_2020',
       'INDE_2020', 'IAA_2020', 'IEG_2020','IPS_2020', 'IDA_2020', 'IPP_2020','IPV_2020', 'IAN_2020', 
       'FASE_2021','TURMA_2021', 'INSTITUICAO_ENSINO_ALUNO_2021','PONTO_VIRADA_2021', 
       'INDE_2021', 'IAA_2021','IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021', 
       'NIVEL_IDEAL_2021','DEFASAGEM_2021', 
       'FASE_2022', 'TURMA_2022','BOLSISTA_2022', 'PONTO_VIRADA_2022',
       'INDE_2022','IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022','IPP_2022', 'IPV_2022', 'IAN_2022', 
       'NIVEL_IDEAL_2022',
       'NOTA_PORT_2022', 'NOTA_MAT_2022', 'NOTA_ING_2022', 'QTD_AVAL_2022',
       'CG_2022', 'CF_2022', 'CT_2022']]

# Retirando alunos da fase 8
df = df[df['FASE_2020'] !=8]

# Filtrando apenas alunos que têm dados para 2020, 2021 e 2022
df_hist = df.drop(columns=['IDADE_ALUNO_2020','ANOS_PM_2020','SINALIZADOR_INGRESSANTE_2021'])
df_hist = df_hist.dropna(subset=['FASE_2020','FASE_2021', 'FASE_2022'])

df_hist['BOLSISTA_2020'] = df_hist['INSTITUICAO_ENSINO_ALUNO_2020'].apply(lambda x: 0 if x == 'Escola Pública' else 1)
df_hist['BOLSISTA_2021'] = df_hist['INSTITUICAO_ENSINO_ALUNO_2021'].apply(lambda x: 0 if x == 'Escola Pública' else 1)
df_hist['BOLSISTA_2022'] = df_hist['BOLSISTA_2022'].apply(lambda x: 0 if x == 'Não' else 1)
df_hist = df_hist[['NOME', 'FASE_2020','FASE_2021','FASE_2022',
                   'BOLSISTA_2020','BOLSISTA_2021','BOLSISTA_2022',
                   'INDE_2020','INDE_2021','INDE_2022', 'IDA_2020','IDA_2021','IDA_2022', 
                   'IEG_2020','IEG_2021','IEG_2022','IAA_2020', 'IAA_2021','IAA_2022',
                   'IPS_2020','IPS_2021','IPS_2022', 'IPP_2020', 'IPP_2021','IPP_2022',
                   'IPV_2020','IPV_2021','IPV_2022','IAN_2020', 'IAN_2021', 'IAN_2022']]

# Filtrando alunos em relação às bolsas
ganhou_bolsa21 = df_hist[(df_hist['BOLSISTA_2020'] == 0) & (df_hist['BOLSISTA_2021'] == 1)]
sempre_bolsa = df_hist[(df_hist['BOLSISTA_2020'] == 1) & (df_hist['BOLSISTA_2021'] == 1) & (df_hist['BOLSISTA_2022'] == 1)]
nunca_bolsa = df_hist[(df_hist['BOLSISTA_2020'] == 0) & (df_hist['BOLSISTA_2021'] == 0) & (df_hist['BOLSISTA_2022'] == 0)]

df_result = melt_df(df_hist)
df_resultG = melt_df(ganhou_bolsa21)
df_resultS = melt_df(sempre_bolsa)
df_resultN = melt_df(nunca_bolsa)

# Machine learning
#(Feature Engineeringg)
df_churn = df.copy()
df_churn['Saída_2021'] = df_churn.apply(lambda row: 
                                         1 if pd.notna(row['FASE_2020']) and pd.isna(row['FASE_2021']) and pd.isna(row['FASE_2022']) else 
                                         (0 if pd.notna(row['FASE_2020']) and pd.notna(row['FASE_2021']) else np.nan), axis=1)
df_churn['Saída_2021'].value_counts()
df_churn['Saída_2022'] = df_churn.apply(lambda row: 
                                         1 if  pd.notna(row['FASE_2021']) and pd.isna(row['FASE_2022']) else 
                                         (0 if pd.notna(row['FASE_2021']) and pd.notna(row['FASE_2022']) else np.nan), axis=1)
df_churn['Saída_2022'].value_counts()

df_saida_21 = df_churn[['NOME','INDE_2020', 'IDA_2020', 'IEG_2020', 'IAA_2020', 'IAN_2020','IPV_2020', 'IPS_2020', 'IPP_2020','Saída_2021']]
df_saida_22 = df_churn[['NOME','INDE_2021', 'IDA_2021', 'IEG_2021', 'IAA_2021', 'IAN_2021','IPV_2021', 'IPS_2021', 'IPP_2021', 'Saída_2022']]

df_saida_21.columns = [col.replace('_2020', '') for col in df_saida_21.columns]
df_saida_21.rename(columns={'Saída_2021':'Saída'}, inplace=True)
df_saida_22.columns = [col.replace('_2021', '') for col in df_saida_22.columns]
df_saida_22.rename(columns={'Saída_2022':'Saída'}, inplace=True)

df_saida_21.dropna(inplace=True)
df_saida_22.dropna(inplace=True)

df_saida = pd.concat([df_saida_21, df_saida_22], ignore_index=True)
indicadores = df_saida[['IDA','IEG', 'INDE']]

#(Modelo)
X = indicadores  
y = df_saida['Saída']  
modelo = pickle.load(open('pred_evasao.pkl','rb'))




# Load: streamlit

apply_custom_css()
# Título da Página
st.title("Análise dos dados da Passos Mágicos")

# Separando as Tabs
tab0, tab1, tab2, tab3= st.tabs(['Geral','Evolução dos Indicadores', 'Previsão de Evasão', 'Resultado'])


with tab0:    
    '''
    ### Sobre a Passos Mágicos

    Essa análise é focada nos alunos em fase escolar, por isso, foram removidos os alunos da fase 8, uma vez que já são 
    universitários e apresentam um perfil distinto dos demais. Selecionando apenas os alunos das fases 0 a 7, separou-se dois 
    grupos diferentes para estudo:

    - **Grupo A:** Alunos que permaneceram na Passos Mágicos durante os três anos analisados (2020, 2021 e 2022);
    - **Grupo B:** Alunos que, em algum momento, deixaram de fazer parte da Passos Mágicos.

    Para o **Grupo A**, foram calculadas as médias dos indicadores em cada ano para identificar o comportamento desses dados 
    ao longo do tempo. Além de calcular as médias gerais dos índices para todos os alunos desse grupo, subdividimos o 
    **Grupo A** em três subgrupos para análises mais detalhadas:

    - Alunos da rede pública durante os três anos;
    - Alunos bolsistas durante os três anos;
    - Alunos que ganharam bolsa em 2021;

    Essa subdivisão permitiu observar os diferentes comportamentos dos índices, ao calcular as médias separadamente para cada 
    subgrupo.

    Para o **Grupo B**, conduzimos uma análise de correlação entre as notas dos diversos índices e o fato de o aluno ter 
    desistido em algum momento ou não.
    '''

with tab1:    
    '''
    ## Análise do Histórico

    ### Histórico Geral dos Alunos
    '''
    st.plotly_chart(plota_indices(df_result,'Médias dos Indicadores dos alunos com histórico entre 2020 e 2022'))
    
    '''  
    A média geral do INDE apresentou um declínio considerável, influenciado diretamente pela queda no IDA e no IEG, que têm 
    maior peso na composição do índice. Além disso, a acentuada redução no IAN e no IPP ao longo dos anos sugere que fatores 
    emocionais e comportamentais estão impactando negativamente o desempenho acadêmico dos alunos.

    Por outro lado, observa-se uma recuperação no IDA e no IEG em 2022, indicando que, apesar das dificuldades enfrentadas, 
    os alunos começaram a retomar seu engajamento nas atividades escolares. Essa recuperação pode estar relacionada ao retorno 
    gradual à normalidade após o período crítico da pandemia, com um ambiente mais favorável ao aprendizado.

    Os indicadores IAA, IPV e IPS, por sua vez, mostraram uma estabilidade relativa ao longo dos três anos, sugerindo que a 
    percepção dos alunos sobre suas capacidades e o impacto da educação em suas vidas não foi profundamente afetada pelas 
    mudanças ocorridas nesse período.

    

    ### Histórico dos subgrupos de alunos 
    '''
    ##st.plotly_chart(plota_indices(df_result,'Evolução Geral dos Indicadores')) ## APAGAR
    st.plotly_chart(plota_indices(df_resultN,'Médias dos Indicadores dos alunos da rede pública'))
    st.plotly_chart(plota_indices(df_resultG,'Médias dos Indicadores dos alunos que ganharam bolsa em 2021'))
    st.plotly_chart(plota_indices(df_resultS,'Médias dos Indicadores dos bolsistas em 2020, 2021 e 2022'))
    
    

    '''
    Os índices médios dos estudantes da rede pública foram, em geral, menores que os dos demais grupos, evidenciando o impacto 
    positivo das bolsas de estudo no desempenho dos alunos. Para os três subgrupos analisados, o __INDE__ apresentou uma queda em 
    2021, o que demonstra que os efeitos da pandemia foram sentidos por todos. No entanto, esse impacto foi mais acentuado 
    entre os alunos da rede pública, o que enfatiza o efeito positivo das bolsas.


    #### IDA e IEG
    O __IEG__ caiu em 2021 para todos os subgrupos, mas recuperou-se em 2022.  Destaca-se o grupo de alunos que ganharam bolsa em 
    2021, que mostrou maior **IEG** do que os demais subgrupos, além de uma queda menos acentuada. Ou seja, :orange[a aquisição da bolsa
    afetou diretamente o engajamento e motivação do aluno].
    
    O __IDA__ também sofreu uma queda em 2021, especialmente entre os alunos da rede pública, cujo desempenho caiu de 
    7.15 para 5.10. Essa diminuição foi o principal fator por trás da redução na média geral do IDA, destacando a 
    vulnerabilidade desses alunos, cujo aprendizado foi bastante prejudicado pela pandemia. No entanto, é notável que esses 
    alunos também apresentaram a :orange[maior recuperação no IDA em 2022]. Isso sugere que, apesar das dificuldades, o apoio 
    oferecido pela Passos Mágicos foi crucial para ajudar os alunos a reverter parte dos impactos adversos. 
    

    #### IAN
    O __IAN__ (Indicador de Adequação de Nível) diminuiu para quase todos os grupos ao longo dos anos, exceto para os alunos 
    que ganharam bolsa em 2021, que mostraram um aumento modesto em 2021 seguido de uma leve queda em 2022. Este fenômeno 
    sugere que os alunos que receberam bolsa estavam mais alinhados com o nível esperado. Enquanto os alunos que apenas na rede
    pública, ou apenas na rede privada nesses três anos, o sugere a necessidade de atenção para os efeitos das
    :orange[estratégias para ajudar os alunos a alcançarem seu nível ideal], especialmente para aqueles que não têm acesso a bolsas.

    #### IAA

    Em 2020, o __IAA__ médio de todos os grupos era próximo de 8.7. Em 2021, o IAA se manteve estável entre os alunos que já tinha bolsa
    em 2020, o que é um sinal positivo. Isso indica que, apesar das dificuldades acadêmicas enfrentadas, esses alunos conseguiram 
    :orange[manter uma visão consistente sobre suas próprias capacidades]. 
    
    Em contraste, os alunos da rede pública e aqueles que receberam bolsa em 2021 experimentaram uma queda no IAA. Ou seja,
    período de pandemia abalou mais a autopercepção desses alunos, destacando a necessidade de :orange[suporte adicional para 
    restaurar a confiança].

    ### IPP

    Ao separar os grupos, observa-se que o __IPP__ permaneceu estável para os alunos com bolsa desde 2020. Isso sugere que, 
    para esses alunos, o desenvolvimento cognitivo e emocional foi mais consistente. O aumento observado na média geral em 2021, 
    seguido por uma queda acentuada em 2022, está mais relacionado aos demais subgrupos. Portanto, é nesses subgrupos que deve 
    haver um :orange[foco maior para o suporte ao desenvolvimento emocional e comportamental] dos alunos.

    '''
    

with tab2:
    '''
    ### Análise da Relação entre Indicadores e a Evasão de Alunos

    Foi realizada uma análise da correlação entre a evasão de alunos e os indicadores de desempenho do ano 
    anterior. Embora a correlação geral seja fraca, os indicadores INDE, IDA e IEG se destacam com correlações 
    mais expressivas em comparação com os demais.
    '''

    st.pyplot(plot_corr(df_saida))
    
    '''
    Com base nesses resultados, foi treinado um modelo de machine learning utilizando regressão logística para prever 
    a evasão de 2020 para 2021 e de 2021 para 2022. As métricas de desempenho do modelo podem ser observada a seguir:  
    
    '''
   
    st.pyplot(evaluate_model(X, y,modelo), use_container_width=True)

    '''
    Avaliando as métricas, nota-se que, para a classe de alunos que não evadiram, o modelo possui uma boa 
    precisão (0.78), o que significa que a maioria das previsões de não evasão estão corretas. No entanto, 
    o recall é mais baixo (0.61), indicando uma maior ocorrência de falsos positivos, o que não é prejudicial 
    para a predição. 

    Já para a classe de alunos evadidos, a precisão indica que apenas 44% das previsões de alunos evadidos estão
    corretas, e isso pode ser explicado pela presença de falsos positivo já identificada. Por outro lado, o recall
    é mais elevado (0.64), mostrando que o modelo é capaz de capturar uma quantidade significativa (64%) dos alunos 
    que efetivamente evadiram. 

    O modelo é moderadamente eficaz para prever a evasão. A acurácia de 0.62 e o f1-score de 0.52 para a classe de 
    evasão (1) sugerem que o modelo pode ser útil como uma ferramenta de apoio na identificação de alunos com 
    maior probabilidade de abandonar, levando em conta os indicadores IDA, IEG e INDE.

    Entendendo das limitações suas limitações e a geração de falsos positivos, o modelo de regressão logística 
    pode ser utilizado para ajudar a antecipar a evasão de alunos, com ressalvas. Podendo assim servir como uma 
    ferramenta inicial para identificar alunos em risco de evasão, que depois devem ser monitorados de perto ou 
    submetidos a avaliações adicionais.


    ### Cálculo do risco de evasão

    A seguir está a ferramenta que é capaz de calcular a previsão segundo o modelo do estudo, a partir dos 
    valores dos indicadores IDA, IEG e INDE.

    '''
    ida = st.number_input('Informe o valor do IDA:', min_value=0.0,max_value=10.0, format="%.2f")
    ieg = st.number_input('Informe o valor do IEG:', min_value=0.0,max_value=10.0, format="%.2f")
    inde = st.number_input('Informe o valor do INDE:', min_value=0.0,max_value=10.0, format="%.2f")

    # Botão para calcular a previsão
    if st.button('Calcular Risco'):
        previsao = prever_evasao(ida, ieg, inde,modelo)
        if previsao == 1:
            st.write('Observação: :red[Risco alto de evasão]')
        else:
            st.write('Observação: :green[Risco menor de evasão]')
with tab3:
    '''
    ###
     - Além de ajudar os alunos a dimunuir a xefasagem a Passos também dá a oportunidade dos alunos a obterem bolsa em
     escola particulares. O que visivelmente contribui para ...
     - Para os alunos da rede publica é preciso dar atenção e impulsionar o alunos para maximizar...
     - E para os alunos que ingressam em escolas particular, é preciso dar surporte para os alunos enfrentarem as dificuldades ...
     - Os indicadores IDA e IEG são muito relevantes para tentar prevenir a desistência dos alunos da Passos Mágicos.
     - Em gersal os alunos com histórco de bolsa se adaptaram bem a nova escola e, além de apresentarem os maiores índices médios,
     também apresentaram as menores quedas. Evidenciando a mudança na vida dessas crianças.
    '''