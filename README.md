# Support Integrity Auditor (SIA)

## Overview

Support Integrity Auditor (SIA) is an AI-powered CRM ticket auditing system designed to detect priority assignment mismatches in customer support tickets.

The system identifies:

* Hidden Crises (high-severity tickets assigned low priority)
* False Alarms (low-severity tickets assigned high priority)
* Consistent Tickets (correctly prioritized)

The project uses self-supervised pseudo-label generation, machine learning classification, and explainable evidence dossiers to audit ticket integrity.

---

## Problem Statement

Customer support teams often assign incorrect priorities to incoming tickets.

Examples:

* Critical issues marked as Low priority
* Minor issues marked as Critical priority

Such mistakes increase response times, customer dissatisfaction, and operational inefficiency.

This project aims to automatically identify these mismatches before they impact service quality.

---

## Features

* CRM ticket analysis
* Pseudo-label generation framework
* Priority mismatch detection
* Hidden Crisis identification
* False Alarm detection
* Evidence dossier generation
* Interactive Streamlit dashboard
* Dataset analytics and visualizations

---

## Tech Stack

### Backend

* Python
* Pandas
* Scikit-learn

### Machine Learning

* TF-IDF Vectorization
* Logistic Regression
* Self-Supervised Pseudo Labeling

### Dashboard

* Streamlit
* Plotly

---

## Project Architecture

Customer Ticket
↓
Pseudo Label Engine
↓
Feature Extraction
↓
ML Classifier
↓
Evidence Dossier
↓
Streamlit Dashboard

---

## Dataset

Customer Support Tickets CRM Dataset

* 20,000 customer support tickets
* Multiple issue categories
* Priority levels
* Resolution metadata
* Customer satisfaction scores

---

## Results

### Baseline Model

* Accuracy: 59.9%
* Macro F1 Score: 0.58

### Mismatch Detection

* Hidden Crises detected
* False Alarms detected
* Consistent ticket identification

---

## Dashboard

The Streamlit dashboard provides:

* KPI metrics
* Mismatch analytics
* Severity heatmaps
* Priority distribution
* Ticket auditing interface
* Downloadable reports

---

## Future Improvements

* Sentence Transformer embeddings
* DeBERTa fine-tuning
* Retrieval-Augmented Evidence Generation
* Real-time CRM integration
* Agent performance analytics

---

## Author

M. Siri Chandana

Civil Engineering, IIT Roorkee
