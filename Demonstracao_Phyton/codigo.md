import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import seaborn as sns
from collections import Counter

class KNNSklearnClassifier:
    def __init__(self):
        # Dados expandidos para melhor demonstração
        self.dados_completos = np.array([
            [1, 2], [2, 3], [3, 6], [6, 9], [7, 10], [8, 8], [9, 2],
            [1.5, 2.5], [2.5, 3.5], [3.5, 6.5], [6.5, 9.5], [7.5, 10.5],
            [8.5, 8.5], [9.5, 2.5], [0.5, 1.5], [4, 7], [5, 8], [8, 3]
        ])
        
        self.classes_completas = [
            'Banana', 'Banana', 'Maçã', 'Maçã', 'Maçã', 'Maçã', 'Banana',
            'Banana', 'Banana', 'Maçã', 'Maçã', 'Maçã', 'Maçã', 'Banana',
            'Banana', 'Maçã', 'Maçã', 'Banana'
        ]
        
        self.cores = {'Maçã': '#e74c3c', 'Banana': '#f1c40f'}
        
        # Modelos sklearn
        self.modelo_knn = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
        self.usar_normalizacao = tk.BooleanVar(value=False)
        
        # Treinar modelo inicial
        self.treinar_modelo()
        
        # Configuração da interface
        self.root = tk.Tk()
        self.root.title("k-NN Classifier com Scikit-learn")
        self.root.geometry("1400x900")
        self.root.configure(bg='#ecf0f1')
        
        # Variáveis de controle
        self.docura_var = tk.DoubleVar(value=5.0)
        self.textura_var = tk.DoubleVar(value=6.0)
        self.k_var = tk.IntVar(value=3)
        self.algoritmo_var = tk.StringVar(value='auto')
        self.metrica_var = tk.StringVar(value='euclidean')
        
        self.setup_ui()
        self.atualizar_calculo()
        
    def treinar_modelo(self):
        """Treina o modelo k-NN com sklearn"""
        X = self.dados_completos.copy()
        y = np.array(self.classes_completas)
        
        if self.usar_normalizacao.get():
            X = self.scaler.fit_transform(X)
        
        self.modelo_knn.fit(X, y)
        
        # Calcular métricas
        y_pred = self.modelo_knn.predict(X)
        self.accuracy = accuracy_score(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.modelo_knn, X, y, cv=5)
        self.cv_mean = cv_scores.mean()
        self.cv_std = cv_scores.std()
        
    def classificar_sklearn(self, novo_ponto):
        """Classifica usando sklearn"""
        ponto = np.array([novo_ponto]).reshape(1, -1)
        
        if self.usar_normalizacao.get():
            ponto = self.scaler.transform(ponto)
        
        # Predição
        classe_predita = self.modelo_knn.predict(ponto)[0]
        
        # Probabilidades (se suportado)
        try:
            probabilidades = self.modelo_knn.predict_proba(ponto)[0]
            prob_dict = dict(zip(self.modelo_knn.classes_, probabilidades))
        except:
            prob_dict = {}
        
        # Vizinhos mais próximos
        X_treino = self.dados_completos.copy()
        if self.usar_normalizacao.get():
            X_treino = self.scaler.transform(X_treino)
        
        distancias, indices = self.modelo_knn.kneighbors(ponto, n_neighbors=self.k_var.get())
        
        vizinhos_info = []
        for i, (dist, idx) in enumerate(zip(distancias[0], indices[0])):
            vizinhos_info.append({
                'indice': idx,
                'distancia': dist,
                'classe': self.classes_completas[idx],
                'ponto': self.dados_completos[idx]
            })
        
        return classe_predita, prob_dict, vizinhos_info
    
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_frame = tk.Frame(main_frame, bg='#2c3e50', relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(title_frame, text="🤖 k-NN Classifier com Scikit-learn", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(title_frame, text="Classificação Profissional usando biblioteca sklearn", 
                                font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack(pady=(0, 5))
        
        academic_label = tk.Label(title_frame, text="Demonstração avançada de Machine Learning", 
                                font=('Arial', 10), fg='#95a5a6', bg='#2c3e50')
        academic_label.pack(pady=(0, 10))
        
        # Frame de conteúdo
        content_frame = tk.Frame(main_frame, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame esquerdo - Gráfico e métricas
        left_frame = tk.Frame(content_frame, bg='#ecf0f1')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Resultado
        result_frame = tk.Frame(left_frame, bg='#1abc9c', relief=tk.RAISED, bd=2)
        result_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.resultado_label = tk.Label(result_frame, text="Classificação: Maçã", 
                                       font=('Arial', 18, 'bold'), fg='white', bg='#1abc9c')
        self.resultado_label.pack(pady=5)
        
        self.confianca_label = tk.Label(result_frame, text="Probabilidade: 85.3%", 
                                       font=('Arial', 14), fg='#ecf0f1', bg='#1abc9c')
        self.confianca_label.pack(pady=(0, 5))
        
        # Métricas do modelo
        metrics_frame = tk.Frame(left_frame, bg='#3498db', relief=tk.RAISED, bd=2)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.accuracy_label = tk.Label(metrics_frame, text="Acurácia: 95.2%", 
                                      font=('Arial', 12, 'bold'), fg='white', bg='#3498db')
        self.accuracy_label.pack(pady=2)
        
        self.cv_label = tk.Label(metrics_frame, text="Cross-Validation: 92.1% (±3.2%)", 
                                font=('Arial', 10), fg='#ecf0f1', bg='#3498db')
        self.cv_label.pack(pady=(0, 5))
        
        # Gráfico
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame direito - Controles avançados
        right_frame = tk.Frame(content_frame, bg='#ecf0f1', width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Notebook para organizar controles
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba: Parâmetros básicos
        basic_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(basic_frame, text="Parâmetros Básicos")
        
        # Slider Doçura
        tk.Label(basic_frame, text="Doçura:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.docura_scale = tk.Scale(basic_frame, from_=0, to=10, resolution=0.1, 
                                    orient=tk.HORIZONTAL, variable=self.docura_var,
                                    command=self.on_change, bg='#ecf0f1', fg='#2c3e50')
        self.docura_scale.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Slider Textura
        tk.Label(basic_frame, text="Textura:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.textura_scale = tk.Scale(basic_frame, from_=0, to=12, resolution=0.1, 
                                     orient=tk.HORIZONTAL, variable=self.textura_var,
                                     command=self.on_change, bg='#ecf0f1', fg='#2c3e50')
        self.textura_scale.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Slider K
        tk.Label(basic_frame, text="Valor de k:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.k_scale = tk.Scale(basic_frame, from_=1, to=15, resolution=1, 
                               orient=tk.HORIZONTAL, variable=self.k_var,
                               command=self.on_change_k, bg='#ecf0f1', fg='#2c3e50')
        self.k_scale.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Aba: Configurações avançadas
        advanced_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(advanced_frame, text="Configurações")
        
        # Algoritmo
        tk.Label(advanced_frame, text="Algoritmo:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        algo_combo = ttk.Combobox(advanced_frame, textvariable=self.algoritmo_var, 
                                 values=['auto', 'ball_tree', 'kd_tree', 'brute'], state='readonly')
        algo_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        algo_combo.bind('<<ComboboxSelected>>', self.on_change_advanced)
        
        # Métrica de distância
        tk.Label(advanced_frame, text="Métrica:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        metric_combo = ttk.Combobox(advanced_frame, textvariable=self.metrica_var, 
                                   values=['euclidean', 'manhattan', 'minkowski', 'chebyshev'], state='readonly')
        metric_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        metric_combo.bind('<<ComboboxSelected>>', self.on_change_advanced)
        
        # Normalização
        norm_check = tk.Checkbutton(advanced_frame, text="Usar normalização dos dados", 
                                   variable=self.usar_normalizacao, command=self.on_change_advanced,
                                   bg='#ecf0f1', fg='#2c3e50')
        norm_check.pack(anchor=tk.W, padx=10, pady=10)
        
        # Botões
        btn_frame = tk.Frame(advanced_frame, bg='#ecf0f1')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btn_frame, text="Retreinar Modelo", command=self.retreinar_modelo,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="Mostrar Relatório", command=self.mostrar_relatorio,
                 bg='#e74c3c', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        # Aba: Resultados
        results_frame = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(results_frame, text="Resultados")
        
        # Tabela de vizinhos
        tk.Label(results_frame, text="Vizinhos Mais Próximos:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.tree_vizinhos = ttk.Treeview(results_frame, columns=('Índice', 'Classe', 'Distância'), 
                                         show='headings', height=6)
        self.tree_vizinhos.heading('Índice', text='Índice')
        self.tree_vizinhos.heading('Classe', text='Classe')
        self.tree_vizinhos.heading('Distância', text='Distância')
        
        self.tree_vizinhos.column('Índice', width=60)
        self.tree_vizinhos.column('Classe', width=80)
        self.tree_vizinhos.column('Distância', width=100)
        
        self.tree_vizinhos.pack(fill=tk.X, padx=10, pady=10)
        
        # Probabilidades
        tk.Label(results_frame, text="Probabilidades:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.prob_text = tk.Text(results_frame, height=4, font=('Arial', 10), 
                                bg='white', fg='#2c3e50')
        self.prob_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Info técnica
        tk.Label(results_frame, text="Informações Técnicas:", font=('Arial', 10, 'bold'), 
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.info_text = tk.Text(results_frame, height=6, font=('Arial', 9), 
                                bg='white', fg='#2c3e50')
        self.info_text.pack(fill=tk.X, padx=10, pady=10)
    
    def on_change(self, *args):
        """Callback para mudanças nos parâmetros básicos"""
        self.atualizar_calculo()
    
    def on_change_k(self, *args):
        """Callback específico para mudança de k"""
        self.modelo_knn.set_params(n_neighbors=self.k_var.get())
        self.atualizar_calculo()
    
    def on_change_advanced(self, *args):
        """Callback para mudanças nas configurações avançadas"""
        self.retreinar_modelo()
    
    def retreinar_modelo(self):
        """Retreina o modelo com novos parâmetros"""
        try:
            # Atualizar parâmetros do modelo
            self.modelo_knn = KNeighborsClassifier(
                n_neighbors=self.k_var.get(),
                algorithm=self.algoritmo_var.get(),
                metric=self.metrica_var.get()
            )
            
            # Retreinar
            self.treinar_modelo()
            
            # Atualizar métricas na interface
            self.accuracy_label.config(text=f"Acurácia: {self.accuracy:.1%}")
            self.cv_label.config(text=f"Cross-Validation: {self.cv_mean:.1%} (±{self.cv_std:.1%})")
            
            # Atualizar cálculo
            self.atualizar_calculo()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao retreinar modelo: {str(e)}")
    
    def atualizar_calculo(self):
        """Atualiza todos os cálculos e visualizações"""
        docura = self.docura_var.get()
        textura = self.textura_var.get()
        
        novo_ponto = [docura, textura]
        
        # Classificar com sklearn
        classe_predita, probabilidades, vizinhos_info = self.classificar_sklearn(novo_ponto)
        
        # Atualizar resultado
        self.resultado_label.config(text=f"Classificação: {classe_predita}")
        
        if probabilidades:
            prob_max = max(probabilidades.values())
            self.confianca_label.config(text=f"Probabilidade: {prob_max:.1%}")
        
        # Atualizar gráfico
        self.atualizar_grafico(novo_ponto, vizinhos_info, classe_predita)
        
        # Atualizar tabela de vizinhos
        self.atualizar_tabela_vizinhos(vizinhos_info)
        
        # Atualizar probabilidades
        self.atualizar_probabilidades(probabilidades)
        
        # Atualizar informações técnicas
        self.atualizar_info_tecnica(novo_ponto, classe_predita)
    
    def atualizar_grafico(self, novo_ponto, vizinhos_info, classe_predita):
        """Atualiza o gráfico de dispersão"""
        self.ax.clear()
        
        # Plotar todos os pontos de treinamento
        for classe in set(self.classes_completas):
            indices = [i for i, c in enumerate(self.classes_completas) if c == classe]
            pontos = self.dados_completos[indices]
            self.ax.scatter(pontos[:, 0], pontos[:, 1], 
                          c=self.cores[classe], label=classe, s=80, alpha=0.6)
        
        # Destacar vizinhos
        if vizinhos_info:
            vizinhos_pontos = [v['ponto'] for v in vizinhos_info]
            vizinhos_array = np.array(vizinhos_pontos)
            self.ax.scatter(vizinhos_array[:, 0], vizinhos_array[:, 1], 
                          s=150, facecolors='none', edgecolors='black', linewidths=2, 
                          label='Vizinhos')
        
        # Plotar novo ponto
        self.ax.scatter(novo_ponto[0], novo_ponto[1], 
                       c=self.cores[classe_predita], s=300, marker='*', 
                       edgecolors='black', linewidths=2, label='Novo Ponto')
        
        # Plotar região de decisão (simplificado)
        self.plotar_regiao_decisao()
        
        self.ax.set_xlabel('Doçura')
        self.ax.set_ylabel('Textura')
        self.ax.set_title(f'k-NN Classifier (k={self.k_var.get()}, {self.metrica_var.get()})')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plotar_regiao_decisao(self):
        """Plota a região de decisão do classificador"""
        try:
            # Criar grid
            x_min, x_max = 0, 10
            y_min, y_max = 0, 12
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                                np.arange(y_min, y_max, 0.5))
            
            # Preparar dados
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            if self.usar_normalizacao.get():
                grid_points = self.scaler.transform(grid_points)
            
            # Predizer para cada ponto do grid
            Z = self.modelo_knn.predict(grid_points)
            
            # Converter classes para números
            class_to_num = {'Banana': 0, 'Maçã': 1}
            Z_num = np.array([class_to_num[z] for z in Z])
            Z_num = Z_num.reshape(xx.shape)
            
            # Plotar região
            self.ax.contourf(xx, yy, Z_num, alpha=0.2, cmap='RdYlBu')
            
        except Exception as e:
            print(f"Erro ao plotar região de decisão: {e}")
    
    def atualizar_tabela_vizinhos(self, vizinhos_info):
        """Atualiza a tabela de vizinhos"""
        # Limpar tabela
        for item in self.tree_vizinhos.get_children():
            self.tree_vizinhos.delete(item)
        
        # Adicionar vizinhos
        for vizinho in vizinhos_info:
            self.tree_vizinhos.insert('', 'end', values=(
                vizinho['indice'],
                vizinho['classe'],
                f"{vizinho['distancia']:.3f}"
            ))
    
    def atualizar_probabilidades(self, probabilidades):
        """Atualiza as probabilidades"""
        self.prob_text.delete(1.0, tk.END)
        
        if probabilidades:
            for classe, prob in probabilidades.items():
                self.prob_text.insert(tk.END, f"{classe}: {prob:.1%}\n")
        else:
            self.prob_text.insert(tk.END, "Probabilidades não disponíveis\n")
            self.prob_text.insert(tk.END, "para este algoritmo/métrica")
    
    def atualizar_info_tecnica(self, novo_ponto, classe_predita):
        """Atualiza informações técnicas"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"""Ponto: ({novo_ponto[0]:.1f}, {novo_ponto[1]:.1f})
Algoritmo: {self.algoritmo_var.get()}
Métrica: {self.metrica_var.get()}
k-vizinhos: {self.k_var.get()}
Normalização: {'Sim' if self.usar_normalizacao.get() else 'Não'}
Dataset: {len(self.dados_completos)} pontos
Classes: {len(set(self.classes_completas))}
Resultado: {classe_predita}"""
        
        self.info_text.insert(tk.END, info)
    
    def mostrar_relatorio(self):
        """Mostra relatório detalhado do modelo"""
        try:
            # Preparar dados
            X = self.dados_completos.copy()
            y = np.array(self.classes_completas)
            
            if self.usar_normalizacao.get():
                X = self.scaler.transform(X)
            
            # Predições
            y_pred = self.modelo_knn.predict(X)
            
            # Gerar relatório
            report = classification_report(y, y_pred, output_dict=True)
            
            # Mostrar em janela
            relatorio_window = tk.Toplevel(self.root)
            relatorio_window.title("Relatório de Classificação")
            relatorio_window.geometry("600x500")
            
            text_widget = tk.Text(relatorio_window, wrap=tk.WORD, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Formatar relatório
            relatorio_str = f"""RELATÓRIO DE CLASSIFICAÇÃO k-NN

Parâmetros do Modelo:
- k-vizinhos: {self.k_var.get()}
- Algoritmo: {self.algoritmo_var.get()}
- Métrica: {self.metrica_var.get()}
- Normalização: {'Sim' if self.usar_normalizacao.get() else 'Não'}

Métricas Gerais:
- Acurácia: {report['accuracy']:.3f}
- Macro Avg Precision: {report['macro avg']['precision']:.3f}
- Macro Avg Recall: {report['macro avg']['recall']:.3f}
- Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}

Métricas por Classe:
"""
            
            for classe in ['Banana', 'Maçã']:
                if classe in report:
                    relatorio_str += f"\n{classe}:"
                    relatorio_str += f"\n  Precision: {report[classe]['precision']:.3f}"
                    relatorio_str += f"\n  Recall: {report[classe]['recall']:.3f}"
                    relatorio_str += f"\n  F1-Score: {report[classe]['f1-score']:.3f}"
                    relatorio_str += f"\n  Support: {report[classe]['support']}"
            
            relatorio_str += f"\n\nCross-Validation (5-fold):"
            relatorio_str += f"\n  Média: {self.cv_mean:.3f}"
            relatorio_str += f"\n  Desvio Padrão: {self.cv_std:.3f}"
            
            text_widget.insert(tk.END, relatorio_str)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar relatório: {str(e)}")
    
    def executar(self):
        """Executa a aplicação"""
        self.root.mainloop()

if __name__ == "__main__":
    app = KNNSklearnClassifier()
    app.executar()
