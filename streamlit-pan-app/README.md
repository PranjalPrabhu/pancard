# Streamlit PAN App

This project is a Streamlit application designed to detect fields on PAN cards and compare images. It integrates functionalities from two main scripts: `finalpan.py` for PAN card field detection and `imagecompare.py` for image comparison.

## Project Structure

```
streamlit-pan-app
├── src
│   ├── finalpan.py       # Detects PAN card fields using YOLO and EasyOCR
│   ├── imagecompare.py    # Compares images based on their content
│   └── app.py            # Main entry point for the Streamlit application
├── requirements.txt       # Lists the required dependencies
└── README.md              # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-pan-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command in your terminal:

```
streamlit run src/app.py
```

Once the application is running, you can:

- Upload an image of a PAN card to detect fields such as the PAN number and name.
- Compare two images to analyze their differences.

## Dependencies

The application requires the following libraries:

- Streamlit
- OpenCV
- YOLO (Ultralytics)
- EasyOCR
- Other libraries as needed in `finalpan.py` and `imagecompare.py`

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.