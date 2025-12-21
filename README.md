# Predicting Short-Term Stock Price Movements Using Machine Learning (Report)

- Category: Application / Empirical Study
- Team Members: Jiho Hahn (jh2982), Ian (icl8)
- Code is in [AML_final_code.ipynb](./AML_final_code.ipynb)

## Abstract

Predicting short-term stock price movements is challenging due to the noisy and non-stationary nature of
financial markets. Yet technical indicators and historical patterns may contain weak but exploitable
predictive structure. In this project, we frame next-day stock direction as a binary classification task and
benchmark logistic regression, XGBoost, and an LSTM model on a multi-ticker U.S. equities dataset
using engineered momentum, volatility, and market-context features. The baseline linear model performs
near random for most tickers, indicating substantial underfitting and strong variation in predictability
across stocks. XGBoost did not significantly improve prediction, while LSTM showed modest
improvements over multiple tickers. No experiments showed accuracy above 60%. These findings
illustrate both the difficulty of short-term financial forecasting and the importance of model flexibility,
feature design, and ticker-specific dynamics in uncovering weak predictive signals.

## Introduction

Stock markets are influenced by many interacting factors, from macroeconomic conditions to
company-specific events, and short-term price movements are often considered unpredictable. Still, past
research and industry practice suggest that certain technical indicators and statistical patterns can
provide useful predictive signals. Our project aims to model daily stock returns as a binary classification
problem (up vs. down) using machine learning techniques introduced in this course.
This problem is motivating because it is both practical and methodologically challenging, since financial
data is noisy, non-stationary, and prone to overfitting. By tackling it, we can assess how well machine
learning methods generalize to a real-world high-variance setting. Also, even small improvements over
random prediction can be valuable in real-world trading and risk management.
This project will be detailed benchmarking / analysis of multiple machine learning models on an existing
financial dataset. We are focused on evaluating their predictive performance and robustness in short-term
stock movement prediction.

## Background

Prior work on short-horizon stock prediction, including studies motivated by the weak-form Efficient
Market Hypothesis, has shown that next-day returns are difficult to predict (Fama). In an earlier milestone
of this project, we trained a logistic regression model as a baseline on the US Stock Market Data &
Technical Indicators dataset (USSMD) from Kaggle. With raw attributes we engineered features to
develop a set of 12 features for our model to train on. We trained and tested per ticker (e.g. AAPL, MSFT,
...) to allow for weight tuning specific to a given stock's technical indicator responses. We also chose to
avoid extreme market volatility, specifically the data in COVID sell-off era which started around mid Feb in
2020.


Results from this earlier logistic regression baseline showed near-random performance on most tickers
and a strong bias toward predicting downward movements for several stocks, indicating that the linear
decision boundary was too restrictive for the nonlinear dynamics of financial time series. The variation in
performance across tickers further suggested that different stocks exhibit different momentum structures
and temporal dependencies that a simple linear model cannot capture. These findings from our baseline
experiment motivated the use of more expressive models like XGBoost which can model nonlinear
interactions between technical indicators, and LSTM which can incorporate temporal structure and
sequential dependencies that daily indicators alone may not fully represent. By introducing these models,
we aim to determine whether greater model flexibility or explicit time-series modeling can extract the weak
predictive signals that the baseline failed to capture.

## Methods

**Dataset and Problem Formulation -** We frame next-day stock price direction as a binary classification
problem (up vs. down) using daily U.S. equity data from the US Stock Market Data & Technical Indicators
dataset on Kaggle. The dataset contains historical OHLC prices, technical indicators, and market context
variables for multiple large-cap stocks. For each ticker, we construct features at day t and predict whether
the close price at day t+1 increases relative to day t. Because financial data is time-dependent, all splits
respect chronological order to prevent information leakage.
**Feature Engineering -** To capture meaningful aspects of short-horizon market behavior, we engineer a
set of 12 indicators commonly used in quantitative trading:

- Price & Returns: Close(t), Return_1d, Return_5d
- Volume: trading activity and liquidity
- Momentum metrics: MA10, MA20, RSI, MACD, MACD_EMA
- Volatility & range: Volatility_10d, High_Low_Range
- Market context: S&P500 1-day return
Indicators are computed per-ticker, aligned by date, and NaNs arising from rolling windows are removed.
Labels are generated by shifting the 1-day return by one step to represent next-day direction. All
experiments are conducted independently for each ticker to reflect stock-specific dynamics.
**Train-Test Split -** For XGBoost experiment, we partitioned the data chronologically using an 80-
time-based split. Specifically, the training dataset comprises the earliest 80% of the time series. For
LSTM, data was split into 80-10-10 ratio which includes validation set for hyperparameter tuning. This
setup simulates the realistic forecasting scenario of training on past data and evaluating on future unseen
data. Periods around early COVID-19 market disruptions are removed to avoid extreme volatility regimes
that could distort learning and evaluation.

## Experiments

We evaluated three classes of models with increasing representational capacity, first being a logistic
regression model as a baseline. This baseline provides a simple benchmark for measuring whether the
engineered indicators contain any linearly separable predictive signal.
The second model we used is XGBoost which is a nonlinear tree-based model, well-suited for tabular
financial features because they naturally capture nonlinear relationships and feature interactions (e.g.,
momentum x volatility). XGBoost also offers strong regularization, reducing the risk of overfitting in noisy


financial data. Hyperparameters such as depth, learning rate, and number of estimators are tuned
through manual small-scale sweeps. We chose a moderate number of trees (200) and a relatively small
learning rate (0.05) to allow the model to gradually fit weak predictive signals while reducing the risk of
overfitting. The tree depth is limited to 4 to restrict model complexity and prevent memorization of
short-term fluctuations that are unlikely to generalize. To further improve robustness, we apply row and
feature subsampling (0.9 each) to inject randomness during training and regularize against correlated
technical indicators. Finally, we use log-loss as the evaluation metric to directly optimize probabilistic
classification performance, which is more informative than accuracy alone in imbalanced and low-signal
settings. Overall, these choices balance expressiveness and regularization, reflecting the weak-signal,
high-noise nature of short-horizon stock movement prediction.
The last model implemented is LSTM. To incorporate temporal dependencies more explicitly, we trained
an LSTM model per ticker using sliding windows of past features (i.e., 30 days) as input sequences which
outputs a binary prediction for the next-day return direction. We used grid search to find the best
hyperparameters among 48 combinations for each ticker. Models with best hyperparameters are then
trained with 20 epochs, optimized with Adam and binary cross-entropy loss.
For evaluation, we used multiple metrics like accuracy, F1-score for the positive (“Up”) class, and
confusion matrices. These metrics are computed per-ticker to highlight variability in predictability among
different stocks.

## Results

_Table 1. Prediction result for Logistic Regression model_

| Ticker | Accuracy | F1-Score | TP  | FP  | FN  | TN  |
|--------|----------|----------|-----|-----|-----|-----|
| TSLA  | 49.30%   | 0.562    | 70  | 64  | 45  | 36  |
| MSFT  | 45.96%   | 0.236    | 60  | 39  | 349 | 270 |
| AMZN  | 54.51%   | 0.699    | 246 | 205 | 7   | 8   |
| GOOGL | 55.03%   | 0.544    | 165 | 118 | 159 | 174 |
| GE    | 48.35%   | 0.486    | 237 | 301 | 200 | 232 |
| GS    | 49.72%   | 0.077    | 15  | 11  | 350 | 342 |
| IBM   | 49.48%   | 0.389    | 156 | 132 | 358 | 324 |
| JPM   | 49.69%   | 0.408    | 168 | 152 | 336 | 314 |
| FB    | 52.09%   | 0.466    | 45  | 36  | 67  | 67  |
| AAPL  | 45.26%   | 0.105    | 23  | 27  | 366 | 302 |

In the logistic regression model’s prediction result shown in Table 1, the majority of tickers exhibited near
random performance with accuracy close to 50%. However, a couple tickers stood out. The result for
(MSFT) was significantly worse than a random guess suggesting overfitting. Additionally, AMZN and
GOOGL showed marginal outperformance which suggests the feature set may have captured some slight
predictive signal specific to those high-growth/tech stocks.
For the XGBoost experiment result shown in Table 2, our implementation did not consistently outperform
the logistic regression baseline across tickers. Accuracy remains close to random guessing for most
stocks, and F1-scores vary substantially by ticker. In addition to accuracy and F1-score, we report AUC to
evaluate the model’s ability to rank positive versus negative days independent of a fixed threshold. While
XGBoost achieves modest AUC improvements for some tickers (e.g., MSFT, GS), most values remain
close to 0.5, reinforcing the limited separability of short-term stock movements. These results indicate that
short-term stock return prediction remains challenging even with more expressive models, and that any


exploitable structure is weak, ticker-specific, and highly sensitive to market dynamics. Overall, the limited
gains over our baseline from XGBoost suggest that model expressiveness alone is insufficient to
overcome the inherent noise and non-stationarity of short-horizon financial data.
Table 3 reports the performance of the LSTM model. Overall, the LSTM provides modest but more
consistent improvements across several tickers, suggesting limited benefit from incorporating temporal
structure. In terms of accuracy, the LSTM outperforms logistic regression on multiple stocks such as
MSFT (57.5% vs. 46.0%), FB (55.7% vs. 52.1%), and GOOGL (55.5% vs. 55.0%), though gains remain
small and many results stay close to chance level. The advantage of the LSTM is more apparent in
F1-score, particularly for MSFT (0.72 vs. 0.24) and GOOGL, indicating a better balance between
precision and recall compared to the baseline, which often exhibits strong bias toward the majority class
and produces excessive false negatives. AUC values for the LSTM are slightly above 0.5 for several
tickers, whereas logistic regression frequently behaves as a near-random classifier, suggesting improved
ranking ability even when classification accuracy remains limited. Nevertheless, for several stocks such
as AMZN and GS, both models fail to extract meaningful signals, underscoring the noisy and weakly
predictable nature of short-term stock movements and highlighting that temporal modeling alone is
insufficient to overcome these challenges.

_Table 2. Prediction result for XGBoost_

| Ticker | Accuracy | F1-Score | AUC  | TP  | FP  | FN  | TN  |
|--------|----------|----------|------|-----|-----|-----|-----|
| TSLA  | 48.60%   | 0.505    | 0.483| 56  | 52  | 58  | 48  |
| MSFT  | 50.56%   | 0.415    | 0.561| 126 | 72  | 283 | 237 |
| AMZN  | 45.06%   | 0.293    | 0.499| 53  | 56  | 200 | 157 |
| GOOGL | 48.14%   | 0.365    | 0.472| 92  | 88  | 232 | 205 |
| GE    | 50.88%   | 0.489    | 0.509| 228 | 267 | 209 | 265 |
| GS    | 51.81%   | 0.433    | 0.526| 132 | 113 | 233 | 240 |
| IBM   | 50.36%   | 0.548    | 0.507| 291 | 259 | 222 | 197 |
| JPM   | 49.95%   | 0.148    | 0.522| 42  | 23  | 462 | 442 |
| FB    | 45.33%   | 0.485    | 0.491| 55  | 60  | 57  | 42  |
| AAPL  | 46.94%   | 0.460    | 0.491| 162 | 154 | 227 | 175 |

_Table 3. Prediction result for LSTM_

| Ticker | Accuracy (%) | F1-Score | AUC  | TP  | FP  | FN  | TN  |
|--------|--------------|----------|------|-----|-----|-----|-----|
| TSLA  | 50.94        | 0.600    | 0.489| 39  | 28  | 24  | 15  |
| MSFT  | 57.54        | 0.720    | 0.490| 195 | 144 | 8   | 11  |
| AMZN  | 47.41        | 0.238    | 0.510| 19  | 20  | 102 | 91  |
| GOOGL | 55.52        | 0.670    | 0.522| 139 | 112 | 25  | 32  |
| GE    | 47.93        | 0.591    | 0.502| 182 | 212 | 40  | 50  |
| GS    | 47.77        | 0.051    | 0.522| 5   | 9   | 178 | 166 |
| IBM   | 54.55        | 0.509    | 0.529| 114 | 75  | 145 | 150 |
| JPM   | 52.48        | 0.547    | 0.535| 139 | 126 | 104 | 115 |
| FB    | 55.66        | 0.591    | 0.571| 34  | 24  | 23  | 25  |
| AAPL  | 50.00        | 0.456    | 0.524| 75  | 56  | 123 | 104 |

Additionally, we conducted an ablation study by removing a certain group of features to measure
importance per signals for conducting better predictions. Removed features are presented using the
subsets in table 4.


_Table 4. Removed features per ablation group_

| Ablation Group | Removed Features |
|---------------|------------------|
| Full Feature Set | No removal (use all 12 features) |
| No Momentum | Remove: MA10, MA20, RSI, MACD, MACD_EMA, Return_5d |
| No Volatility | Remove: Volatility_10d, High_Low_Range |
| No Market Context | Index_SP500_Return_1d |

_Table 5. Mean F1 scores across tickers from XGBoost prediction_

| Ablation Group       | Mean F1 |
|---------------------|---------|
| Full Feature Set    | 0.414   |
| No Momentum         | 0.450   |
| No Volatility       | 0.401   |
| No Market Context   | 0.400   |

_Table 6. Mean F1 scores across tickers from LSTM prediction_

| Ablation Group       | Mean F1 |
|---------------------|---------|
| Full Feature Set    | 0.497   |
| No Momentum         | 0.470   |
| No Volatility       | 0.473   |
| No Market Context   | 0.440   |

The ablation results from table 5 and 6 show clear differences in how models leverage feature groups.
Logistic regression performs near random with the full feature set (mean F1 = 0.3972), confirming its
limited capacity for modeling nonlinear financial dynamics. XGBoost provides only modest improvement
(F1 = 0.414), and surprisingly performs best when momentum features are removed (F1 = 0.45),
suggesting these indicators may add noise rather than signal for tree-based models. In contrast, the
LSTM achieves the highest performance (F1 = 0.497) and degrades under all ablations, with the largest
drop when market context is removed, indicating the importance of both temporal structure and broader
market signals. Overall, the results highlight that feature utility is model-dependent and that sequential
models are better suited for extracting weak predictive signals in short-term stock data.

## Conclusion

In this project, we investigated the feasibility of predicting next-day stock price direction using a range of
machine learning models under realistic, time-ordered evaluation settings. We learned that
short-to-medium-horizon financial prediction is challenging, and carefully engineered technical and
market-context features contain weak predictive signals. Among the models evaluated, logistic regression
consistently underperformed, highlighting its inability to capture nonlinear interactions and temporal
dependencies. XGBoost, despite its greater expressiveness, provided only marginal improvements and
often failed to outperform the linear baseline, suggesting that nonlinear feature interactions alone are
insufficient in this setting. The LSTM model achieved the most consistent (though still modest) gains,
particularly in F1-score, indicating that explicitly modeling temporal structure helps mitigate class
imbalance and capture weak sequential patterns present in some tickers. However, overall predictive
performance remained close to chance for many stocks, underscoring key limitations of our approach,
including limited feature diversity, per-ticker data scarcity, sensitivity to market conditions, and the
exclusion of extreme volatility periods. For future work, incorporating additional information sources such
as news or social-media sentiment, macroeconomic indicators, and explicit market regime detection could
improve robustness. More expressive sequence models, including transformer-based architectures, may
better capture long-range dependencies and cross-feature interactions. Importantly, predictive
performance is likely to vary across market regimes (e.g., high- versus low-volatility periods), so
regime-aware or adaptive modeling frameworks can be a promising direction for improving short-term 
stock prediction.

## References

- Kohli, Nikhil. US Stock Market Data & Technical Indicators. Kaggle, 2020,
https://www.kaggle.com/datasets/nikhilkohli/us-stock-market-data-60-extracted-features
- Fama, Eugene F. “Efficient Capital Markets: A Review of Theory and Empirical Work.” _The Journal of
Finance_ , vol. 25, no. 2, 1970, pp. 383-417. _JSTOR_ , https://doi.org/10.2307/2325486.


