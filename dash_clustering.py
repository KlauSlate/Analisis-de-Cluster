# los primero es importar las librerias
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import scipy.cluster.hierarchy as shc
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# codigo principal para la implementacion del dash
def main():
    global data
    inertia_scores = []
    silueta_scores = []
    st.markdown('# Proyecto Final Analitica de datos')
    uploaded_file = st.sidebar.file_uploader('Seleccionar archivo',type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    st.sidebar.markdown('Selecciones el tipo de analisis')
    step = st.sidebar.radio('', ['Introduccion', 'Analisis Exploratorio', 'Resultados de cluster', 'Reduccion Dimensionalidad'])
    if step == 'Introduccion':
        run_introduccion()
    elif step == 'Analisis Exploratorio':
        run_features()  
    elif step == 'Resultados de cluster':
        run_clustering()
    elif step == 'Reduccion Dimensionalidad':
        run_dimensionalidad()
    
    
def run_features():
    st.markdown('### **Analisis Exploratorio de Datos (EDA)**')
    st.write('Vista previa de los datos:')
    st.write(data.head())
    col1, col2, col3 = st.columns([2,1,1])
    col1.write(data.dtypes)
    col3.write(data.groupby('Gender')['Gender'].count())
    var_x = st.selectbox('Variable para el eje x', data.columns)
    var_y = st.selectbox('Variable para el eje y', data.columns)
    var_z = st.selectbox('Variable para especificar', data.columns)
    fig = px.scatter(data, x=var_x, y=var_y, title=f'Scatter plot de {var_x} vs {var_y}', color=var_z, color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    
    fig2 = plt.figure(figsize=(8,5))
    sns.heatmap(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(), annot=True, cmap='coolwarm')
    plt.title('Diagrama de calor para ver las correlaciones de las variables')
    st.pyplot(fig2)
    
    var_x_1 = st.selectbox('Seleccione la variable', data.columns[2:5])
    fig3 = px.histogram(data, x=data[var_x_1], nbins=137)
    st.plotly_chart(fig3)
    
    st.markdown('### **Calculo de los coeficientes**')

    X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    x_1 = np.arange(2, 13)
    inertia_scores = []
    silueta_scores = []
    for n_clusters in range(2,13):
        kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto').fit(X)
        silueta = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto').fit_predict(X)
        inertia_scores.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X, silueta)
        silueta_scores.append(silhouette_avg)
    fig4 = px.line(x=x_1, y= inertia_scores, title='Diagrama de codo', markers=True)
    fig4.add_vline(x=3, line_width=2, line_dash='dot', line_color='MediumPurple')
    fig4.add_vline(x=5, line_width=2, line_dash='dashdot', line_color='LightSeaGreen')
    st.plotly_chart(fig4)

    fig5 = px.line(x=x_1, y= silueta_scores, title='Promedio puntaje de silueta', markers=True)
    fig5.add_vline(x=5, line_width=2, line_dash='dashdot', line_color='LightSeaGreen')
    st.plotly_chart(fig5)

def run_clustering():
    st.markdown('### **Analisis de clustering K-Means**')
    k_cluster = st.sidebar.slider('Elija un numero de clusters', min_value=2, max_value=10, step=1)
    
    # aplicando el algoritmo k-means
    var_x = st.selectbox('Variable para el eje x', data.columns)
    var_y = st.selectbox('Variable para el eje y', data.columns)
    clustering_kmeans = KMeans(n_clusters=k_cluster, random_state=10, n_init='auto').fit(data[[var_x, var_y]])
    data_kmeans = data.copy()
    data_kmeans['Cluster_Kmeans'] = clustering_kmeans.labels_
    fig3 = px.scatter(data_kmeans, x=var_x, y=var_y, title=f'Scatter plot de {var_x} vs {var_y}', 
                     color=clustering_kmeans.labels_.astype(str))
    st.plotly_chart(fig3)
    
    st.markdown('### **Analisis de clustering Jerarquico**')
    escalado = st.selectbox('Quiere escalar los datos???', ['Si', 'No'])
    var_independientes = st.multiselect('Selecciones las variables a clusterizar', list(['Annual Income (k$)', 'Spending Score (1-100)', 'Age']))
    if escalado == 'Si':
        df_scaled = normalize(data[var_independientes])
        #dend = shc.dendrogram(shc.linkage(df_scaled), method='ward')
        fig4 = ff.create_dendrogram(df_scaled)
        st.plotly_chart(fig4)
        data_jerarquico = data.copy()
        cluster_jerarquico = AgglomerativeClustering(n_clusters=k_cluster, metric='euclidean', linkage='average')
        cluster_jerarquico.fit_predict(df_scaled)
        data_jerarquico['Cluster_Jerarquico'] = cluster_jerarquico.labels_
        fig5 = px.scatter(data_jerarquico, x=var_x, y=var_y, title=f'Scatter plot de {var_x} vs {var_y}',
                          color=cluster_jerarquico.labels_.astype(str))
        st.plotly_chart(fig5)
    elif escalado == 'No':
        fig6 = ff.create_dendrogram(data[var_independientes])
        st.plotly_chart(fig6)
        data_jerarquico_2 = data.copy()
        cluster_jerarquico_2 = AgglomerativeClustering(n_clusters=k_cluster, metric='euclidean', linkage='average')
        cluster_jerarquico_2.fit_predict(data_jerarquico_2[var_independientes])
        data_jerarquico_2['Cluster_Jerarquico_2'] = cluster_jerarquico_2.labels_
        fig7 = px.scatter(data_jerarquico_2, x=var_x, y=var_y, title=f'Scatter plot de {var_x} vs {var_y}',
                          color=cluster_jerarquico_2.labels_.astype(str))
        st.plotly_chart(fig7)

def run_dimensionalidad():
    st.markdown('# **Reduccion de dimensionalidad**')
    array_x = data[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
    nums = range(1,4)
    var_ratio = []
    for num in nums:
        pca = PCA(n_components=num).fit(array_x)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))
    
    x_pca = pca.transform(array_x)
    fig = px.line(x=nums, y=var_ratio, title='Numero de componentes v/s Varianza', markers=True)
    fig.update_xaxes(title_text='Numero de componentes')
    fig.update_yaxes(title_text='Varianza')
    st.plotly_chart(fig)
    
    k_cluster = st.slider('Elija un numero de clusters', min_value=2, max_value=10, step=1)
    
    # aplicando el algoritmo k-means
    clustering_kmeans = KMeans(n_clusters=k_cluster, random_state=10, n_init='auto').fit(x_pca)
    data_kmeans = data.copy()
    data_kmeans['Cluster_Kmeans'] = clustering_kmeans.labels_
    fig3 = px.scatter(x_pca, x=x_pca[:,0], y=x_pca[:,1], title=f'Scatter plot de Componente1 vs Componente2', 
                     color=clustering_kmeans.labels_.astype(str))
    st.plotly_chart(fig3)
        
    # aplicando el algoritmo jerarquico
    data_jerarquico_2 = data.copy()
    cluster_jerarquico_2 = AgglomerativeClustering(n_clusters=k_cluster, metric='euclidean', linkage='average')
    cluster_jerarquico_2.fit_predict(data_jerarquico_2[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])
    data_jerarquico_2['Cluster_Jerarquico_2'] = cluster_jerarquico_2.labels_
    fig7 = px.scatter(x_pca, x=x_pca[:,0], y=x_pca[:,1], title=f'Scatter plot de Componente1 vs Componente2',
                        color=cluster_jerarquico_2.labels_.astype(str))
    st.plotly_chart(fig7)
    
    
    
    
def run_introduccion():
    st.write('aca va la introduccion')

if __name__ == '__main__':
    main()    
    