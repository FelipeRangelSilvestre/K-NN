# k-NN Classifier com Scikit-learn

Uma aplicação interativa para demonstração e experimentação com o algoritmo k-Nearest Neighbors (k-NN) usando Python, Tkinter e Scikit-learn.

## 📋 Descrição

Este projeto implementa uma interface gráfica completa para classificação usando o algoritmo k-NN, permitindo visualizar em tempo real como diferentes parâmetros afetam o desempenho do classificador. A aplicação utiliza um dataset de exemplo com frutas (Maçã e Banana) baseado em características como doçura e textura.

## ✨ Funcionalidades

### 🎯 Classificação Interativa
- **Predição em tempo real**: Arraste os sliders para ver a classificação instantânea
- **Visualização gráfica**: Gráfico de dispersão com regiões de decisão
- **Destaque de vizinhos**: Visualização dos k-vizinhos mais próximos

### 🔧 Parâmetros Configuráveis
- **Valor de k**: Número de vizinhos (1-15)
- **Algoritmo**: auto, ball_tree, kd_tree, brute
- **Métrica de distância**: euclidean, manhattan, minkowski, chebyshev
- **Normalização**: Opção para normalizar os dados

### 📊 Métricas e Análises
- **Acurácia do modelo**
- **Cross-validation** (5-fold)
- **Relatório de classificação** completo
- **Probabilidades** de cada classe
- **Tabela de vizinhos** mais próximos

### 🎨 Interface Avançada
- **Design profissional** com cores organizadas
- **Abas organizadas** para diferentes funcionalidades
- **Visualizações dinâmicas** com matplotlib
- **Informações técnicas** detalhadas

## 🚀 Instalação

### Pré-requisitos
```bash
Python 3.7+
```

### Dependências
```bash
pip install tkinter matplotlib numpy scikit-learn pandas seaborn
```

### Instalação das dependências
```bash
# Usando pip
pip install -r requirements.txt

# Ou instalar individualmente
pip install matplotlib numpy scikit-learn pandas seaborn
```

## 📦 Estrutura do Projeto

```
knn-classifier/
│
├── knn_classifier.py          # Arquivo principal
├── README.md                  # Este arquivo
├── requirements.txt           # Dependências
└── screenshots/              # Capturas de tela (opcional)
```

## 🎮 Como Usar

### 1. Executar a aplicação
```bash
python knn_classifier.py
```

### 2. Interface Principal
A aplicação possui três abas principais:

#### **Parâmetros Básicos**
- **Doçura**: Ajusta a característica doçura (0-10)
- **Textura**: Ajusta a característica textura (0-12)
- **Valor de k**: Define o número de vizinhos

#### **Configurações Avançadas**
- **Algoritmo**: Escolha o algoritmo de busca
- **Métrica**: Selecione a métrica de distância
- **Normalização**: Ative/desative a normalização dos dados
- **Retreinar Modelo**: Aplica as novas configurações
- **Mostrar Relatório**: Exibe métricas detalhadas

#### **Resultados**
- **Vizinhos Mais Próximos**: Tabela com os vizinhos encontrados
- **Probabilidades**: Probabilidades de cada classe
- **Informações Técnicas**: Detalhes da configuração atual

### 3. Interpretação dos Resultados

#### Gráfico Principal
- **Pontos coloridos**: Dataset de treinamento
- **Estrela**: Ponto a ser classificado
- **Círculos pretos**: k-vizinhos mais próximos
- **Fundo colorido**: Regiões de decisão

#### Métricas
- **Acurácia**: Percentual de acertos no dataset
- **Cross-Validation**: Validação cruzada com 5 folds
- **Relatório**: Precision, Recall e F1-Score por classe

## 🔬 Algoritmos Disponíveis

### Algoritmos de Busca
- **auto**: Escolha automática baseada nos dados
- **ball_tree**: Eficiente para dados de alta dimensão
- **kd_tree**: Rápido para baixa dimensão
- **brute**: Força bruta (sempre funciona)

### Métricas de Distância
- **euclidean**: Distância euclidiana clássica
- **manhattan**: Distância de Manhattan (L1)
- **minkowski**: Generalização das anteriores
- **chebyshev**: Distância de Chebyshev (L∞)

## 📊 Dataset

O dataset inclui 18 pontos com duas características:
- **Doçura** (0-10): Nível de doçura da fruta
- **Textura** (0-12): Textura da superfície

**Classes:**
- 🍌 **Banana**: Pontos amarelos
- 🍎 **Maçã**: Pontos vermelhos

## 🎯 Casos de Uso

### 🎓 Educacional
- Ensino de algoritmos de Machine Learning
- Demonstração de conceitos k-NN
- Visualização de hiperparâmetros

### 🔬 Pesquisa
- Prototipagem rápida de classificadores
- Teste de diferentes configurações
- Análise de sensibilidade de parâmetros

### 💼 Profissional
- Baseline para projetos de classificação
- Demonstração para stakeholders
- Validação de conceitos

## 🛠️ Personalização

### Modificar o Dataset
```python
# Em __init__
self.dados_completos = np.array([
    # Seus dados aqui
])

self.classes_completas = [
    # Suas classes aqui
]
```

### Adicionar Novas Métricas
```python
# Em setup_ui, adicionar à lista de métricas
values=['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'sua_metrica']
```

### Personalizar Cores
```python
# Em __init__
self.cores = {
    'Classe1': '#cor1',
    'Classe2': '#cor2'
}
```

## 📈 Métricas Explicadas

### **Acurácia**
Percentual de classificações corretas sobre o total.

### **Cross-Validation**
Validação cruzada com 5 folds para estimar a performance real.

### **Precision**
Proporção de predições positivas corretas.

### **Recall**
Proporção de casos positivos identificados corretamente.

### **F1-Score**
Média harmônica entre Precision e Recall.

## 🐛 Troubleshooting

### Erro de Importação
```bash
# Instalar dependências faltantes
pip install matplotlib numpy scikit-learn pandas seaborn
```

### Tkinter não encontrado
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

### Problemas de Performance
- Reduza o número de pontos no dataset
- Use algoritmo 'ball_tree' para dados maiores
- Desative a plotagem da região de decisão

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- **Desenvolvimento**: Demonstração acadêmica de Machine Learning
- **Inspiração**: Algoritmo k-NN clássico da literatura

## 🙏 Agradecimentos

- Scikit-learn pela excelente biblioteca de ML
- Matplotlib pela visualização de dados
- Tkinter pela interface gráfica

## 📚 Referências

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [k-NN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Python Machine Learning](https://www.python.org/)

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**
