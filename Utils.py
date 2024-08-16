import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import streamlit as st

# Função para ajustar o layout
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Reduzindo as margens laterais */
        .css-18e3th9 {
            padding: 1rem 1rem 1rem 1rem;
        }
        
        /* Ajustando o cabeçalho */
        .css-1d391kg {
            padding: 1rem 1rem 1rem 1rem;
        }

        /* Ajusta a largura dos blocos de conteúdo */
        .block-container {
            padding: 1rem 2rem 1rem 2rem;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

def melt_df(df_in):
    melted = pd.melt(df_in, id_vars=['NOME'], 
                        value_vars=['INDE_2020', 'INDE_2021', 'INDE_2022', 
                                    'IDA_2020', 'IDA_2021', 'IDA_2022',
                                    'IEG_2020', 'IEG_2021', 'IEG_2022',
                                    'IAA_2020', 'IAA_2021', 'IAA_2022',
                                    'IPS_2020', 'IPS_2021', 'IPS_2022',
                                    'IPP_2020', 'IPP_2021', 'IPP_2022',
                                    'IPV_2020', 'IPV_2021', 'IPV_2022',
                                    'IAN_2020', 'IAN_2021', 'IAN_2022'],
                        var_name='Metric_Year', value_name='Value')

    melted[['Metric', 'Year']] = melted['Metric_Year'].str.split('_', expand=True)
    melted = melted.drop(columns=['Metric_Year'])
    result = melted.groupby(['Year', 'Metric']).mean(numeric_only=True).unstack()
    result.columns = result.columns.droplevel()
    return result

def plota_indices(result,title):

    # Configurando a figura com 3 subplots em uma linha
    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=("IDA e IEG", "IAA e IAN", "IPV, IPS e IPP")
    )

    # Subplot 1: INDE, IDA, IEG 
    fig.add_trace(go.Scatter(x=result.index, y=result['INDE'], mode='lines+markers', name='INDE', marker=dict(color='orange'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result['IDA'], mode='lines+markers', name='IDA', marker=dict(color='red'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result['IEG'], mode='lines+markers', name='IEG', marker=dict(color='darkred'), showlegend=False), row=1, col=1)

    # Subplot 2: INDE, IAA, IPV, IAN 
    fig.add_trace(go.Scatter(x=result.index, y=result['INDE'], mode='lines+markers', name='INDE', marker=dict(color='orange'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=result.index, y=result['IAA'], mode='lines+markers', name='IAA', marker=dict(color='blue'), showlegend=False), row=1, col=2)    
    fig.add_trace(go.Scatter(x=result.index, y=result['IAN'], mode='lines+markers', name='IAN', marker=dict(color='deepskyblue'), showlegend=False), row=1, col=2)

    # Subplot 3: INDE, IPS, IPP 
    fig.add_trace(go.Scatter(x=result.index, y=result['INDE'], mode='lines+markers', name='INDE', marker=dict(color='orange'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=result.index, y=result['IPV'], mode='lines+markers', name='IPV', marker=dict(color='DeepPink'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=result.index, y=result['IPS'], mode='lines+markers', name='IPS', marker=dict(color='mediumpurple'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=result.index, y=result['IPP'], mode='lines+markers', name='IPP', marker=dict(color='pink'), showlegend=False), row=1, col=3)

    # Rotulando os índices
    # Subplot 1 (INDE, IDA, IEG)
    fig.add_annotation(x=result.index[-1], y=result['INDE'].iloc[-1], text='INDE', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='orange'), row=1, col=1)
    fig.add_annotation(x=result.index[-1], y=result['IDA'].iloc[-1], text='IDA', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='red'), row=1, col=1)
    fig.add_annotation(x=result.index[-1], y=result['IEG'].iloc[-1], text='IEG', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='darkred'), row=1, col=1)

    # Subplot 2 (INDE, IAA, IAN)
    fig.add_annotation(x=result.index[-1], y=result['INDE'].iloc[-1], text='INDE', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='orange'), row=1, col=2)
    fig.add_annotation(x=result.index[-1], y=result['IAA'].iloc[-1], text='IAA', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='dodgerblue'), row=1, col=2)    
    fig.add_annotation(x=result.index[-1], y=result['IAN'].iloc[-1], text='IAN', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='deepskyblue'), row=1, col=2)

    # Subplot 3 (INDE, IPV, IPS, IPP)
    fig.add_annotation(x=result.index[-1], y=result['INDE'].iloc[-1], text='INDE', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='orange'), row=1, col=3)
    fig.add_annotation(x=result.index[-1], y=result['IPV'].iloc[-1], text='IPV', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='DeepPink'), row=1, col=3)
    fig.add_annotation(x=result.index[-1], y=result['IPS'].iloc[-1], text='IPS', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='mediumpurple'), row=1, col=3)
    fig.add_annotation(x=result.index[-1], y=result['IPP'].iloc[-1], text='IPP', showarrow=False, xanchor='left', yanchor='middle', font=dict(color='pink'), row=1, col=3)

    # Ajustando os eixos e layout
    fig.update_yaxes(range=[5, 9], title_text='Valor Médio do Índice', row=1, col=1)
    fig.update_yaxes(range=[5, 9], row=1, col=2)
    fig.update_yaxes(range=[5, 9], row=1, col=3)
    fig.update_xaxes(tickmode='array', tickvals=result.index)

   
    # Ajustando o layout
    fig.update_layout(
        height=400, 
        width=1200, 
        title_text=title, 
        title_x=0.5, 
        title_xanchor='center',
        title_y=0.95,
        showlegend=False,  # Removendo a legenda
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True)
    )

    return fig

def plot_corr(df_saida):
    matriz_corr = df_saida.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 2))
    sns.heatmap(matriz_corr[['Saída']].transpose(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlação entre Indicadores e Saída dos alunos')
    
    plt.tight_layout()
    
    return fig

def evaluate_model(X, y,modelo_treinado):

    # Dividindo o dataset em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # Balanceamento das classes
    smote = SMOTE(random_state=31)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = modelo_treinado
    model.fit(X_train_resampled, y_train_resampled)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo com a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Mostrar acurácia e relatório de classificação
    st.write(f"Acurácia da previsão: {accuracy:.2f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.drop(columns=['support'], inplace = True)
    df_report = df_report.round(2)

    st.dataframe(df_report.head(2))
    
    st.write("\nMatriz Confusão:")

    fig, ax = plt.subplots(figsize=(3,3))  # Aumente ligeiramente o tamanho para evitar overlap
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 10})  # Reduzindo o tamanho da fonte

    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Previsto')
    plt.tight_layout() 

    return fig


def prever_evasao(ida, ieg, inde, modelo_treinado):
    # Cria um array com os valores inseridos pelo usuário
    input_data = np.array([[ida, ieg, inde]])
    # Faz a previsão usando o modelo carregado
    previsao = modelo_treinado.predict(input_data)
    return previsao[0]