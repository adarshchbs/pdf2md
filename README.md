# PDF to Markdown Conversion 

Welcome to our PDF to Markdown Conversion GitHub repository! This service is designed to facilitate an easy transition from PDF documents (that are searchable) to structured Markdown files. The primary goal is to prepare the content in a way that it becomes friendly for further processing and integration with Large Language Models (LLMs) such as ChatGPT, Ollama, etc.

## Features

- **PDF to Markdown Conversion**: Convert your searchable PDF documents into clean, structured Markdown (.md) files.
- **Table Content Removal**: Efficiently removes text inside bounded tables to ensure the consistency and readability of the output text. It detect bounded tables with great accuracy.
- **Structured Output**: Generates a Markdown file that is more structured, making it optimized for use as input to LLMs or for any other needs that require structured documentation.

## Prerequisites

Before you begin using this service, ensure that your PDF files are searchable. Currently, our service **does not support OCR (Optical Character Recognition)**, so the PDFs must be inherently text-based and searchable.

## Quick Start
### Docker
Clone the github repo. Then run docker compose.

`docker compose up`

The service will run on port 8080.


## Demo

### UI
![alt text](image.png)

### Ouput
![alt text](image-1.png)


## Upcoming Features

- Output the bounded table as Markdown table.
- Support OCR

## Contributing

Contributions to enhance the PDF to Markdown Conversion are more than welcome. Please feel free to fork the repository, make improvements, and submit pull requests. We're looking forward to your innovative ideas and contributions.
