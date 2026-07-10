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
| Accuracy  | 86,94% |
| Precision | 86,98% |
| Recall    | 86,94% |

O treinamento utilizou **15 épocas** com early stopping (paciência de 5 épocas). A melhor performance foi atingida na **época 7**, com restauração automática dos melhores pesos.

#### Análise de Confusão

Os pares de caracteres com maior índice de confusão:

| Par (Verdadeiro → Predito) | Ocorrências |
|----------------------------|-------------|
| 'f' → 'F'                  | 152         |
| 'L' → '1'                  | 149         |
| 'O' → '0'                  | 129         |
| 'q' → '9'                  | 120         |
| 'F' → 'f'                  | 111         |
| 'I' → '1'                  | 94          |

A confusão entre caracteres visualmente semelhantes ('f'/'F', 'L'/'1', 'O'/'0') é esperada dada a similaridade morfológica inerente, acentuada pela variabilidade da escrita manuscrita no dataset EMNIST.

---

## Aplicação

> A ser implementada na próxima fase.
