# Handwritten Character Recognition

Sistema de reconhecimento de caracteres manuscritos utilizando Rede Neural Convolucional (CNN) treinada sobre o dataset EMNIST.

---

## Treinamento

### Ambiente de Execução

O treinamento foi realizado no **Google Colab** com acesso à GPU **NVIDIA T4**. A integração foi feita diretamente pelo **VS Code**, conectando o kernel remoto do Colab ao ambiente de desenvolvimento local.

A escolha pelo Colab se deu pela disponibilidade imediata de GPU sem necessidade de configuração local de CUDA/cuDNN, acelerando significativamente o ciclo de treino-evaluação.

### Download do Modelo

Como o Colab roda em um ambiente efêmero, o arquivo `emnist_model.keras` gerado após o treinamento é automaticamente baixado para a máquina local através de uma célula no final do notebook que dispara o download via navegador.

### Estrutura Modular

O código de treinamento foi modularizado em separação de responsabilidades:

```text
training/
├── main.ipynb          # Pipeline completo: treino, avaliação, exportação
├── config.py           # Hiperparâmetros, arquitetura e labels
├── libs/
│   ├── data_loader.py  # Download e pré-processamento do EMNIST
│   ├── model.py        # Construção e compilação da CNN
│   ├── trainer.py      # Loop de treinamento com early stopping
│   └── evaluator.py    # Métricas, matriz de confusão, análise
├── models/             # Modelos .keras salvos (gitignored)
└── requirements.txt    # Dependências
```

### Resultados do Treinamento

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 86,91% |
| Precision | 87,36% |
| Recall    | 86,91% |

O treinamento utilizou **15 épocas** com early stopping (paciência de 5 épocas). A melhor performance foi atingida na **época 14**, com restauração automática dos melhores pesos.

#### Análise de Confusão

### Quais são as letras que apresentam maior índice de confusão pelo modelo?

Com base na análise de confusão, os 15 pares com maior índice de confusão são:

| Par (Verdadeiro → Predito) | Ocorrências |
|----------------------------|-------------|
| 'F' → 'f'                  | 199         |
| 'L' → '1'                  | 199         |
| '0' → 'O'                  | 130         |
| 'I' → '1'                  | 113         |
| 'O' → '0'                  | 98          |
| 'q' → '9'                  | 87          |
| '9' → 'q'                  | 76          |
| '1' → 'I'                  | 63          |
| 'L' → 'I'                  | 61          |
| 'f' → 'F'                  | 59          |
| 'g' → 'q'                  | 54          |
| '2' → 'Z'                  | 45          |
| 'g' → '9'                  | 44          |
| '5' → 'S'                  | 35          |
| '9' → 'g'                  | 30          |

### Quais são as hipóteses teóricas que justificam essas dificuldades de classificação?

As dificuldades de classificação do modelo para os pares identificados podem ser justificadas por diversas hipóteses teóricas, principalmente relacionadas à **similaridade visual** e **variações tipográficas**:

1. **Similaridade Visual:** Muitas das letras e números confundidos possuem formas visuais muito parecidas, especialmente em certas fontes ou quando a escrita manual não é perfeitamente padronizada. Exemplos claros são:
   - **'L' e '1'**: Em muitas fontes, a letra 'L' maiúscula e o número '1' são quase idênticos, distinguindo-se apenas por pequenos traços ou a ausência deles.
   - **'0' (zero) e 'O' (letra O maiúscula)**: São frequentemente indistinguíveis visualmente, sendo um problema clássico em reconhecimento de caracteres.
   - **'I' (letra I maiúscula) e '1' (número um)**: Assim como 'L' e '1', a similaridade visual é muito alta.
   - **'F' e 'f'**: Embora uma seja maiúscula e a outra minúscula, ambas compartilham características estruturais que podem levar à confusão, especialmente em fontes manuscritas onde a 'f' minúscula pode ter um laço ou traço que a assemelha à 'F' maiúscula.
   - **'q' e '9'**: A letra 'q' minúscula e o número '9' têm formas arredondadas com um traço descendente ou ascendente que pode ser interpretado erroneamente.

2. **Variabilidade na Escrita Manual:** A EMNIST é baseada em caracteres manuscritos. A caligrafia individual introduz uma grande variabilidade na forma como um mesmo caractere é escrito. Pequenas diferenças na inclinação, espessura da linha, tamanho e proporção podem fazer com que um caractere se assemelhe a outro, levando a classificações incorretas.

3. **Características Compartilhadas e Ruído:** Algoritmos de CNN buscam padrões e características (edges, curvas, etc.). Se dois caracteres compartilham muitas características visuais, o modelo pode ter dificuldade em aprender as pequenas distinções que os separam. O ruído inerente aos dados ou a baixa resolução das imagens também pode exacerbar essa dificuldade, obscurecendo as nuances entre caracteres semelhantes.

4. **Balanceamento do Dataset (em menor grau neste caso):** Embora o dataset EMNIST Balanced seja projetado para ter um número igual de amostras por classe, se houver um desequilíbrio sutil ou se algumas amostras forem de qualidade inferior para classes específicas, isso pode impactar a capacidade do modelo de distinguir caracteres semelhantes.

Em resumo, a principal hipótese é que a **ambiguidade visual inerente** a certos pares de caracteres, acentuada pela **variabilidade da escrita manual**, representa um desafio significativo para o modelo, independentemente da sua arquitetura.

---

## Aplicação

> A ser implementada na próxima fase.
