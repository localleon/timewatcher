# Timewatcher 3000

## Overview

Timewatcher 3000 is a Streamlit application designed to analyze and visualize your work hours from SAP Zeitnachweis PDF files. The app provides various insights into your work patterns, including total hours worked, overtime, most productive weeks, and more.

## Features

- **PDF Parsing**: Extracts data from SAP Zeitnachweis PDF files.
- **Data Cleanup**: Cleans and preprocesses the extracted data.
- **Data Analysis**: Provides insights into your work patterns, including:
  - Total hours worked
  - Overtime analysis
  - Most and least active days
  - Work-life balance score
  - Most productive weeks
  - Peak log-off times
  - Average start times
  - Favorite work types
  - Long weekends
  - Vacation analysis

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd timewatcher
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run main.py
    ```

2. Upload your SAP Zeitnachweis PDF files using the file uploader in the app.

3. View the analysis and insights provided by the app.

## File Structure

- `main.py`: The main Streamlit application file.
- `.gitignore`: Git ignore file to exclude unnecessary files from the repository.
- `README.md`: This file.

## Dependencies

- camelot
- pandas
- matplotlib
- numpy
- streamlit
- wordcloud

## License

This project is licensed under the MIT License.