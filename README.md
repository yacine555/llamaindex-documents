# RAG Architecture with Llama-Index to retrieve and query documents

## Description

RAG architecture with web scrapping and Node Postprocessor for document retrieval using the [Llamaindex](https://www.llamaindex.ai/) Data Framework

## Getting Started

Follow these instructions to run the app on your local machine for development and testing purposes.

### Prerequisites and Dependencies

Before you begin, ensure you have the following installed:
- Python 3.10.10 or later. Note that this was only tested on 3.10.10
- [Pipenv](https://pipenv.pypa.io/en/latest/) 


Here are the PIP modules used

- [**python-dotenv (1.0.0)**](https://pypi.org/project/python-dotenv/1.0.0/): Reads key-value pairs from a `.env` file and sets them as environment variables.
- [**Llama-index (0.9.10)**](https://pypi.org/project/llama-index/0.9.10/): Provides a GPT Index as a data framework for  LLM application.
- [**streamlit (1.29.0)**](https://pypi.org/project/streamlit/1.29.0/): An app framework for Machine Learning and Data Science to create apps quickly.
- [**NTLK (3.8.1)**](https://www.nltk.org/) [pip](https://pypi.org/project/streamlit/1.29.0/): Natural Language Toolkit is a leading platform for building Python programs to work with human language data.


### Installation

Recommend using Install pipenv or other vitual environment
```bash
pipenv install
pipenv shell
pipenv --version
python --version
```

Clone the repository and install the required packages:

```bash
git clone https://github.com/yacine555/llamindex-documents
cd llamindex-helloworld
pipenv install -r requirements.txt 
```


### Setting Up

To use Hugging Face and OpenAi services, you need to sign up for their APIs and set your API keys:

```bash
export OPENAI_API_KEY='your_langchain_api_key'
export PINECONE_API_KEY='your_langchain_api_key'
```

Check the LLM embeding size and update the variable


### Running the Application

Start the application by running:

```bash
pipenv run python main.py
```
or



Run the the app streamlit
```bash
pipenv run streamlit run main.py
```



## Deployment

Notes on how to deploy the application in a live environment.

## Built With

- [Framework](#) - The web framework used.
- [Database](#) - Database system.
- [Others](#) - Any other frameworks, libraries, or tools used.

## Contributing

Guidelines for contributing to the project.

[CONTRIBUTING.md](CONTRIBUTING.md)

## Versioning

Information on versioning system used, typically [SemVer](http://semver.org/).

## Authors

- **Name** - *Initial work* - [Profile](#)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Resources
- [Llamaindex RAGconcept](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)

## Acknowledgments

