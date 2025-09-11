# LoginGuard-Intelligence

An AI-powered dashboard that detects suspicious user login attempts using the Isolation Forest machine learning model.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Features](#features)
4. [Architecture & Workflow](#architecture--workflow)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Datasets](#datasets)
8. [Model](#model)
9. [Web Application (Dashboard)](#web-application-dashboard)
10. [Future Work](#future-work)
11. [Contributors](#contributors)
12. [License](#license)

## Problem Statement

User accounts are often targeted via unauthorized / suspicious login attempts. It’s challenging for system administrators to manually detect out-of-norm login behaviours (sudden location changes, odd time of access, etc.). You need a system that automatically flags potential login threats so that they can be investigated or blocked.

## Solution Overview

LoginGuard-Intelligence takes login data (features such as user ID, login time, location, maybe device, etc.), uses an **unsupervised anomaly detection model** (Isolation Forest) to identify suspicious / anomalous login attempts, and presents these in an interactive dashboard. The dashboard lets users (admins) view flagged logins, see metrics, and possibly investigate further.


## Features

* Automated detection of suspicious login attempts.
* Visual dashboard to view flagged anomalies.
* Summary statistics: number of logins, number of flagged/malicious ones, by time, by user, etc.
* Interactive filtering (by user, time, maybe device/location).
* Easy to deploy (Streamlit‐based web app).

## Architecture & Workflow

1. **Data collection**: Input login records (e.g. timestamp, user, IP location/device).
2. **Preprocessing**: Clean / format data (e.g. missing values, categorical features → encoding, timestamp → useful features).
3. **Modeling**: Use Isolation Forest to learn what “normal” logins look like continuously, then flag those that differ significantly.
4. **Dashboard / Visualization**: Use Streamlit to build UI that shows flagged logins, visual summaries.
5. **User workflow**: Admin checks dashboard, investigates flagged cases, takes action.

## Installation

To run this project locally:

```bash
# clone repo
git clone https://github.com/Anushkad134/LoginGuard-Intelligence.git
cd LoginGuard-Intelligence

# optionally create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

## Usage

To launch the app locally:

```bash
streamlit run app.py

Then navigate to `http://localhost:8501` (or whatever Streamlit gives you) in your browser. You will see:

* A home screen / dashboard.
* A panel for uploaded or stored login data.
* A view of flagged suspicious logins.
* Filters and statistics.


## Datasets

* The sample dataset `login_data.csv` is included for demonstration purposes.
* It contains fields such as \[list relevant columns: e.g. user\_id, timestamp, IP, location, device, maybe success/failure].
* You can replace with your own login logs (similar schema) to use the model.


## Model

* **Isolation Forest**: unsupervised learning algorithm to detect anomalies.
* Trained on “normal” login data; flags points that are outliers.
* Hyperparameters (e.g., number of trees, contamination rate) are set in code; maybe you allow tuning.


## Web Application (Dashboard)

* Built using **Streamlit**.
* Interface allows admin to:
  • View total login attempts.
  • View number / list of suspicious ones.
  • Filter by time / user / maybe other features.
  • See summary charts (e.g. logins by hour, flagged by time).
* Deployed online (link).


## Future Work

* Add more features (e.g. device risk, geolocation, velocity / impossible travel detection).
* Use supervised learning (if labelled data exists).
* Real-time monitoring instead of batch.
* Integrate alerting (email / SMS / push) for flagged logins.
* More robust front-end / UI improvements.



