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
    ## Sobre a Passos Mágicos
    A Passos Mágicos é uma associação dedicada a utilizar a educação como uma ferramenta de transformação para 
    crianças e jovens em situação de vulnerabilidade social no município de Embu-Guaçu - SP. A associação 
    desenvolveu um índice sintético abrangente que integra uma série de avaliações sobre o progresso educacional 
    de seus alunos. Esse índice é composto por diversos indicadores cuidadosamente selecionados, que refletem 
    os princípios e valores da instituição. Esses indicadores permitem medir o impacto das ações 
    psicopedagógicas realizadas, fornecendo uma análise detalhada do desenvolvimento educacional de cada aluno.

    A fórmula abaixo ilustra como o INDE (Índice de Desenvolvimento Educacional) é calculado, sendo composto 
    por múltiplos indicadores que contribuem para uma visão holística do progresso acadêmico:'''
    st.latex(r'''INDE = 0.2 \cdot IDA + 0.2 \cdot IEG + 0.1 \cdot IAN + 0.1 \cdot IAA + 0.2 \cdot IPV + 0.1 \cdot IPS + 0.1 \cdot IPP''')

    '''
    Os indicadores que compõem o INDE são os seguintes:

    - __IDA__: Indicador de Desempenho acadêmico;
    - __IEG__: Indicador de Engajamento;
    - __IAN__: Indicador de Adequação de Nível;
    - __IAA__: Indicador de Autoavaliação;
    - __IPV__: Indicador de Ponto de Virada;
    - __IPS__: Indicador Psicossocial;
    - __IPP__: Indicador Psicopedagógico.

    
    ## Sobre a análise dos dados

    Foram disponibilizados dados anonimizados de alunos nos anos de 2020, 2021 e 2022. 
    Esta análise foca exclusivamente nos alunos em idade escolar, removendo aqueles da Fase 8 (universitários), 
    cujos perfis diferem dos demais. A análise concentra-se, portanto, 
    nos alunos das Fases 0 a 7, os quais foram divididos em dois grupos principais, cada um abordado de maneira 
    distinta:
    - __Grupo A:__ Alunos que permaneceram na Passos Mágicos ao longo dos três anos analisados (2020, 2021 e 2022)
    - __Grupo B:__ Alunos que, em algum momento, deixaram de fazer parte da Passos Mágicos.

    Para o __Grupo A__, a abordagem foi mais analítica. Foram calculadas as médias dos indicadores educacionais 
    para cada ano, com o objetivo de 
    identificar tendências e mudanças no comportamento desses dados ao longo do tempo. Além disso, o grupo 
    foi subdividido em três subgrupos para uma análise ainda mais profunda:

    - Alunos da rede pública ao longo todos os três anos;
    - Alunos que obtiveram bolsa em 2021;
    - Alunos bolsistas durante todo o período.
    

    Essa subdivisão permite identificar padrões específicos entre os diferentes perfis, proporcionando uma visão 
    mais detalhada do impacto das condições socioeconômicas e das oportunidades educacionais nos resultados dos 
    alunos.

    Já para o __Grupo B__, a análise foi conduzida com uma abordagem estatística, utilizando técnicas de machine 
    learning, especificamente com um modelo de regressão logística. O foco foi explorar a relação entre os 
    diversos indicadores avaliados e a evasão da Passos Mágicos. O objetivo central foi prever e identificar os 
    alunos com maior risco
    de evasão, permitindo a implementação de estratégias preventivas direcionada a esses estudantes. 


    '''
    st.markdown("---")
    st.markdown(
            """
            <style>
            .small-font {
             font-size: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True
    )
    st.markdown(
        '<span class="small-font"> Análise elaborada por Nycolle Nailde de O. B. Pontes </span>',
        unsafe_allow_html=True
        )

with tab1:    
    '''
    ## Histórico Geral dos Alunos
    '''
    st.plotly_chart(plota_indices(df_result,'Médias dos Indicadores dos alunos com histórico entre 2020 e 2022'))
    
    '''  
    A média geral do INDE apresentou um declínio considerável, influenciado principalmente pelas quedas no IDA e no IEG, que têm 
    maior peso na composição desse índice. Além disso, houve uma acentuada redução no IAN e no IPP ao longo dos anos, sugerindo 
    que fatores emocionais e comportamentais podem estar impactando negativamente o desempenho acadêmico dos alunos.

    Por outro lado, observa-se uma recuperação no IDA e no IEG em 2022, indicando que, apesar das dificuldades enfrentadas, 
    os alunos começaram a retomar seu engajamento nas atividades escolares. Essa recuperação pode estar relacionada ao retorno 
    gradual à normalidade após o período crítico da pandemia, criando um ambiente mais favorável ao aprendizado.

    Os indicadores IAA, IPV e IPS, por outro lado, mantiveram-se relativamente estáveis ao longo dos três anos. Isso sugere que 
    :orange[a percepção dos alunos da Passos Mágicos sobre suas capacidades e o impacto da educação em suas vidas não foi profundamente 
    afetada pelas mudanças ocorridas no período de pandemia].

    

    ### Histórico dos subgrupos de alunos 
    '''
    ##st.plotly_chart(plota_indices(df_result,'Evolução Geral dos Indicadores')) ## APAGAR
    st.plotly_chart(plota_indices(df_resultN,'Médias dos Indicadores dos alunos da rede pública'))
    st.plotly_chart(plota_indices(df_resultG,'Médias dos Indicadores dos alunos que ganharam bolsa em 2021'))
    st.plotly_chart(plota_indices(df_resultS,'Médias dos Indicadores dos bolsistas em 2020, 2021 e 2022'))
    
    

    '''
    Quando analisamos os diferentes subgrupos, os alunos da rede pública apresentaram, em geral, indicadores médios 
    inferiores aos dos demais grupos, o que evidencia o impacto positivo das bolsas de estudo no desempenho acadêmico. 
    Foi observada uma queda no __INDE__ em 2021, o que demonstra que os efeitos da pandemia foram sentidos por todos. No entanto, 
    esse impacto foi mais acentuado 
    entre os alunos da rede pública, o que enfatiza outro :orange[efeito positivo das bolsas].


    ### IDA e IEG
    O __IEG__ caiu em 2021 para todos os subgrupos, mas mostrou recuperação em 2022.  Destaca-se o grupo de alunos que ganharam bolsa em 
    2021, que apresentou maior engajamento (IEG) do que os demais grupos, além de uma queda menos acentuada. Ou seja, :orange[a concessão 
    da bolsa teve um impacto positivo direto no engajamento e na motivação desses alunos].
    
    O __IDA__ também sofreu uma queda em 2021, particularmente entre os alunos da rede pública, cujo desempenho acadêmico caiu de 
    7.15 para 5.10. Essa queda foi o principal fator responsável pela redução na média geral do IDA, destacando a 
    vulnerabilidade desses alunos, cujo aprendizado foi bastante prejudicado pela pandemia. Contudo, é notável que esses 
    alunos também apresentaram a :orange[maior recuperação no IDA em 2022]. Isso sugere que, :orange[o apoio 
    oferecido pela Passos Mágicos foi essencial para ajudar os alunos a superar parte dos impactos negativos do período da pandemia]. 
    

    ### IAN
    O __IAN__ (Indicador de Adequação de Nível) mostrou uma tendência de queda para quase todos os grupos ao longo dos anos, com 
    exceção dos alunos que receberam bolsa em 2021, que apresentaram um aumento modesto no indicador naquele ano, seguido de uma 
    leve queda em 2022. Este fato sugere que os alunos que obtiveram bolsa estavam mais alinhados com o nível esperado. Para os 
    alunos exclusivamente da rede pública ou privada durante os três anos, os dados apontam a necessidade de maior :orange[atenção 
    e foco nas estratégias para ajudar os alunos a alcançarem seu nível adequado], especialmente para aqueles que não possuem a bolsa.

    ### IAA

    Em 2020, o __IAA__ médio de todos os grupos era próximo de 8.7. Em 2021, o indicador manteve-se estável entre os alunos com
    bolsa em 2020, um sinal positivo de que, apesar das dificuldades acadêmicas enfrentadas (queda de IDA e IEG), :orange[os bolsistas
    da Passos Mágicos conseguiram manter uma visão consistente sobre suas próprias capacidades]. 
    
    Em contraste, os alunos da rede pública apresentaram uma queda no __IAA__ de 8.7 para 8.0. Ou seja,
    período de pandemia afetou mais a autopercepção desse grupo, enfatizando a necessidade de :orange[suporte adicional para 
    restaurar a confiança] desses alunos.

    ### IPP

    Analisando os subgrupos, observa-se que o __IPP__ permaneceu estável para os alunos com bolsa desde 2020, indicando que :orange[o 
    efeito a longo prazo da bolsa em escola particular favoreceu o desenvolvimento cognitivo e emocional mais consistente]. 
    O aumento observado na média geral do IPP em 2021, seguido por uma queda acentuada em 2022, está mais relacionado aos estudantes da 
    rede pública e aos estudantes que adquiram bolsa mais recentemente. 

    '''
    st.markdown("---")
    st.markdown(
            """
            <style>
            .small-font {
             font-size: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True
    )
    st.markdown(
        '<span class="small-font"> Análise elaborada por Nycolle Nailde de O. B. Pontes </span>',
        unsafe_allow_html=True
        )

with tab2:
    '''
    ## Análise da Relação entre Indicadores e a Evasão de Alunos

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
    corretas, e isso pode ser explicado pela presença de falsos positivos já identificada. Por outro lado, o recall
    é mais elevado (0.64), mostrando que o modelo é capaz de capturar uma quantidade significativa (64%) dos alunos 
    que efetivamente evadiram. 

    O modelo é moderadamente eficaz para prever a evasão. A acurácia de 0.62 e o f1-score de 0.52 para a classe de 
    evasão (1) sugerem que o modelo pode ser útil como uma ferramenta de apoio na identificação de alunos com 
    maior probabilidade de abandonar, levando em conta os indicadores IDA, IEG e INDE.

    Entendendo das limitações suas limitações e a geração de falsos positivos, o modelo de regressão logística 
    pode ser utilizado para ajudar a antecipar a evasão de alunos, com ressalvas. Podendo assim servir como uma 
    ferramenta inicial para identificar alunos em risco de evasão, que depois devem ser monitorados de perto ou 
    submetidos a avaliações adicionais.


    ## Cálculo do risco de evasão

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
            st.write(':red[__Risco alto de evasão__]')
        else:
            st.write(':green[__Risco menor de evasão__]')

    st.markdown("---")
    st.markdown(
            """
            <style>
            .small-font {
             font-size: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True
    )
    st.markdown(
        '<span class="small-font"> Análise elaborada por Nycolle Nailde de O. B. Pontes </span>',
        unsafe_allow_html=True
        )
with tab3:
    '''
    ## Contribuições deste trabalho

    A Passos Mágicos desempenha um papel essencial no desenvolvimento acadêmico, cognitivo e emocional de seus alunos. 
    O acompanhamento contínuo e a concessão de bolsas de estudo em escolas particulares têm mostrado impactos extremamente positivos, 
    ajudando a mitigar os desafios trazidos pela pandemia e outros fatores externos. Esses benefícios 
    ficaram ainda mais evidentes em 2022, quando muitos alunos conseguiram recuperar o desempenho e engajamento escolar após o período 
    desafiador da pandemia. A concessão de bolsas tem sido um diferencial na vida dos alunos, e isso é refletido nos indicadores 
    consistentemente superiores apresentados pelos bolsistas em comparação aos alunos da rede pública.

    Além disso, o estudo de machine learning apresentou uma ferramenta promissora para a identificação precoce dos alunos em risco de 
    evasão. Isso permite que a Passos Mágicos adote intervenções direcionadas, monitorando de perto os alunos que apresentam sinais de 
    vulnerabilidade, garantindo assim que mais alunos possam se beneficiar do acompanhamento e do apoio fornecido pela associação.

    :orange[__A ferramenta de previsão e as estratégias de suporte pode ajudar a Passos Mágicos a continuar a transformar vidas, garantindo que cada 
    aluno tenha a chance de prosperar tanto acadêmica quanto pessoalmente__].

    '''

    st.markdown("---")
    st.markdown(
            """
            <style>
            .small-font {
             font-size: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True
    )
    st.markdown(
        '<span class="small-font"> Análise elaborada por Nycolle Nailde de O. B. Pontes </span>',
        unsafe_allow_html=True
        )