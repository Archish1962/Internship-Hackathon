# AI Powered Educational System

A Streamlit-based web application for analyzing text and code, providing comprehensive metrics on writing quality, sentiment, grammar, code syntax, logic, and more. Ideal for students, educators, and developers looking to improve their writing or programming skills.

## Features
- **Text Analysis**:
  - Sentiment analysis using VADER and RoBERTa models
  - Readability metrics (Flesch Reading Ease, Kincaid Grade Level)
  - Grammar and style checks (run-on sentences, passive voice, weak words)
  - Vocabulary richness and text structure analysis
  - Subject-specific misconception detection (e.g., math, programming)
  - Comparison with reference text for accuracy and overlap
- **Code Analysis**:
  - Syntax checking for Python code
  - Logic, efficiency, and style evaluation (PEP 8 compliance)
  - Detailed feedback with improvement suggestions
- **Comparison Mode**:
  - Side-by-side analysis of text and code
- **Batch Analysis**:
  - Analyze multiple text files with comparative visualizations
- **Interactive UI**:
  - Custom-styled interface with metric cards, tabs, and Plotly visualizations
  - Code editor with syntax highlighting
  - Session management for tracking analysis history
- **Configurable**:
  - Select analysis mode, subject, topic, feedback style, and visualization options

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Access the App**:
   - Open your browser and navigate to `http://localhost:8501` after running the app.
2. **Select Analysis Mode**:
   - Choose "Text Analysis," "Code Analysis," or "Comparison Mode" from the sidebar.
3. **Text Analysis**:
   - Enter text, upload a `.txt` file, or upload multiple files for batch analysis.
   - Configure subject (e.g., Math, Programming) and topic for tailored feedback.
   - View results in tabs: Writing Quality, Sentiment, Statistics, Issues, Insights.
4. **Code Analysis**:
   - Input Python code in the code editor.
   - Provide problem context (optional) for better logic evaluation.
   - Review syntax, logic, efficiency, and style scores with feedback.
5. **Comparison Mode**:
   - Analyze text and code side-by-side with comparative visualizations.
6. **Customize**:
   - Adjust feedback style (Encouraging, Detailed, Concise) and toggle visualizations.
   - Clear session data using the sidebar button.

## Dependencies
See `requirements.txt` for a list of required Python packages.

## Project Structure
- `app.py`: Main Streamlit application code
- `requirements.txt`: List of dependencies
- `README.md`: Project documentation

## Notes
- The application currently supports Python code analysis only.
- Some features (e.g., RoBERTa sentiment analysis) may require significant memory or GPU support.
- Ensure NLTK resources are downloaded (handled automatically on first run).

## Contributing
Contributions are welcome! Please submit issues or pull requests to the repository.
