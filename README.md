# 🔍 k-NN Classifier - Classificação de Frutas

Uma aplicação web interativa que demonstra o algoritmo k-Nearest Neighbors (k-NN) para classificação de frutas baseada em características como doçura e textura.

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como trabalho acadêmico para a disciplina de Álgebra Linear, demonstrando a aplicação prática de conceitos como:
- Distância Euclidiana
- Classificação por vizinhança
- Visualização de dados em 2D

## 🎯 Funcionalidades

### Interface Interativa
- **Controles deslizantes** para ajustar doçura e textura
- **Seleção do valor k** (número de vizinhos)
- **Visualização em tempo real** das mudanças

### Visualizações
- **Gráfico de dispersão** mostrando todos os pontos de treinamento
- **Destaque dos vizinhos** mais próximos
- **Ponto de teste** com formato de estrela
- **Cores diferenciadas** para cada classe de fruta

### Informações Detalhadas
- **Tabela de distâncias** calculadas para todos os pontos
- **Contagem de votos** dos vizinhos mais próximos
- **Porcentagem de confiança** da classificação
- **Avisos automáticos** para valores pares de k

## 🚀 Como Usar

1. **Abra o arquivo HTML** em qualquer navegador moderno
2. **Ajuste os controles**:
   - **Doçura**: Deslize para alterar o valor (0-10)
   - **Textura**: Deslize para alterar o valor (0-10)
   - **Valor k**: Escolha o número de vizinhos (1-7)
3. **Observe os resultados**:
   - Classificação predita
   - Confiança da predição
   - Visualização gráfica atualizada
   - Detalhes dos cálculos

## 📊 Dados de Exemplo

O sistema utiliza um conjunto de dados sintético com 7 pontos de treinamento:

| Ponto | Doçura | Textura | Classe |
|-------|---------|---------|---------|
| A     | 1.0     | 2.0     | Banana  |
| B     | 2.0     | 3.0     | Banana  |
| C     | 3.0     | 6.0     | Maçã    |
| D     | 6.0     | 9.0     | Maçã    |
| E     | 7.0     | 10.0    | Maçã    |
| F     | 8.0     | 8.0     | Maçã    |
| G     | 9.0     | 2.0     | Banana  |

## 🔧 Tecnologias Utilizadas

- **HTML5**: Estrutura da página
- **CSS3**: Estilização moderna com gradientes e animações
- **JavaScript**: Lógica do algoritmo k-NN
- **Chart.js**: Visualização gráfica dos dados
- **Design Responsivo**: Compatível com diferentes tamanhos de tela

## 📈 Algoritmo k-NN

### Como Funciona

1. **Cálculo de Distâncias**: Calcula a distância euclidiana entre o ponto de teste e todos os pontos de treinamento
   ```
   d = √[(x₁-x₂)² + (y₁-y₂)²]
   ```

2. **Seleção de Vizinhos**: Escolhe os k pontos mais próximos

3. **Classificação**: A classe mais comum entre os vizinhos é atribuída ao ponto de teste

4. **Confiança**: Calculada como a porcentagem de vizinhos da classe vencedora

### Características Importantes

- **Valores ímpares de k**: Recomendados para evitar empates
- **Sensibilidade aos dados**: O algoritmo é sensível a outliers
- **Simplicidade**: Não requer treinamento prévio
- **Interpretabilidade**: Fácil de entender e explicar

## 🎨 Design e UX

- **Gradientes modernos**: Visual atrativo e profissional
- **Animações suaves**: Transições em controles e hover effects
- **Cores intuitivas**: Vermelho para maçãs, amarelo para bananas
- **Layout responsivo**: Adapta-se a diferentes dispositivos
- **Feedback visual**: Destacamento dos vizinhos selecionados

## 📱 Responsividade

O layout se adapta automaticamente para:
- **Desktop**: Grade de duas colunas
- **Tablet/Mobile**: Layout de coluna única
- **Controles touch-friendly**: Sliders otimizados para toque

## 🔍 Detalhes Técnicos

### Estrutura do Código
```javascript
// Principais funções
- calcularDistancia(p1, p2)     // Distância euclidiana
- atualizar()                   // Recalcula e atualiza visualizações
- atualizarGrafico()           // Atualiza o gráfico Chart.js
- atualizarTabelaDistancias()  // Mostra distâncias calculadas
- atualizarVotos()             // Exibe contagem de votos
```

### Dependências
- Chart.js 3.9.1 (via CDN)
- Navegador com suporte a ES6+

## 🎓 Objetivos Educacionais

Este projeto demonstra:
- **Conceitos de Álgebra Linear** na prática
- **Algoritmos de Machine Learning** básicos
- **Visualização de dados** interativa
- **Desenvolvimento web** moderno
- **Interface usuário** intuitiva

## 📝 Possíveis Melhorias

- Adição de mais características (dimensões)
- Implementação de outras métricas de distância
- Carregamento de datasets personalizados
- Exportação de resultados
- Comparação com outros algoritmos
- Validação cruzada
- Normalização de dados

## 🤝 Contribuição

Este é um projeto educacional. Sugestões de melhorias são bem-vindas:
1. Funcionalidades adicionais
2. Melhorias na interface
3. Otimizações de performance
4. Correções de bugs

## 📄 Licença

Projeto desenvolvido para fins educacionais - livre para uso acadêmico.

## 👨‍💻 Desenvolvedor

Desenvolvido para a disciplina de Álgebra Linear como demonstração prática do algoritmo k-NN usando conceitos de distância euclidiana.

---

**🎯 Experimente agora!** Abra o arquivo HTML em seu navegador e explore como o algoritmo k-NN classifica frutas baseado em suas características!
