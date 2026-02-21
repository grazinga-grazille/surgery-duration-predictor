# Surgery Duration Predictor 🏥

A machine learning application that predicts surgical procedure duration using a Random Forest model trained on historical OR cases from a Canadian hospital.

Built as part of research conducted in the Department of Management Sciences at the **University of Waterloo**.

---

## Live Demo

> **[Launch the Streamlit App](https://surgery-duration-predictor-qlxrwbxqovglitrsclpi2a.streamlit.app/)

---

## Features

- **Predict** surgery duration from procedure description, room, specialty, and priority
- **Compare** model prediction against originally booked duration
- **Explore** model performance metrics and top feature importances
- **Interactive** sidebar inputs with instant results

---

## How It Works

| Step | Details |
|------|---------|
| **Input** | Procedure description (free text), OR room, specialty, patient type, surgical priority |
| **Text features** | TF-IDF bigrams → TruncatedSVD (150 components, ~91.6% variance) |
| **Categorical features** | One-hot encoded: patient type, room, specialty |
| **Model** | Random Forest Regressor (100 trees, max depth 15) |
| **Target** | log(actual duration in minutes) — exponentiated at prediction time |

---

## Author

**Gatik Gola** · MSCI 623 · University of Waterloo
