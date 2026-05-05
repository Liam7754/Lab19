"""
Tech Company Corpus.

A small but structurally rich corpus that exercises multi-hop reasoning:
companies, their founders, subsidiaries, products, investors, acquisitions,
locations, and rivalries. The corpus is intentionally split across many
short documents so Flat RAG often retrieves only one fragment, while
GraphRAG can traverse links between fragments.
"""

DOCUMENTS = [
    {
        "id": "doc_openai_1",
        "text": (
            "OpenAI was founded by Sam Altman and Elon Musk in 2015 in San "
            "Francisco. The company's mission is to ensure that artificial "
            "general intelligence benefits all of humanity."
        ),
    },
    {
        "id": "doc_openai_2",
        "text": (
            "OpenAI develops the GPT family of large language models, "
            "including GPT-4 and GPT-4o. ChatGPT is OpenAI's flagship "
            "consumer product, released in November 2022."
        ),
    },
    {
        "id": "doc_openai_3",
        "text": (
            "Microsoft is a major investor in OpenAI, having invested over "
            "ten billion dollars. Microsoft integrates OpenAI's models into "
            "its Copilot products."
        ),
    },
    {
        "id": "doc_anthropic_1",
        "text": (
            "Anthropic was founded by Dario Amodei and Daniela Amodei in "
            "2021. Both founders previously worked at OpenAI. Anthropic is "
            "headquartered in San Francisco."
        ),
    },
    {
        "id": "doc_anthropic_2",
        "text": (
            "Anthropic builds the Claude family of large language models. "
            "Claude is positioned as a competitor to ChatGPT and Gemini."
        ),
    },
    {
        "id": "doc_anthropic_3",
        "text": (
            "Google and Amazon are investors in Anthropic. Amazon committed "
            "up to four billion dollars in investment."
        ),
    },
    {
        "id": "doc_google_1",
        "text": (
            "Google was founded by Larry Page and Sergey Brin in 1998 at "
            "Stanford University. Google's headquarters is in Mountain View, "
            "California."
        ),
    },
    {
        "id": "doc_google_2",
        "text": (
            "Alphabet is the parent company of Google, formed in 2015. "
            "Sundar Pichai is the CEO of both Google and Alphabet."
        ),
    },
    {
        "id": "doc_google_3",
        "text": (
            "DeepMind is a subsidiary of Alphabet, acquired by Google in "
            "2014. DeepMind is based in London and was founded by Demis "
            "Hassabis."
        ),
    },
    {
        "id": "doc_google_4",
        "text": (
            "Google develops the Gemini family of large language models. "
            "Gemini competes directly with ChatGPT and Claude."
        ),
    },
    {
        "id": "doc_microsoft_1",
        "text": (
            "Microsoft was founded by Bill Gates and Paul Allen in 1975. "
            "Microsoft is headquartered in Redmond, Washington. Satya "
            "Nadella is the current CEO."
        ),
    },
    {
        "id": "doc_microsoft_2",
        "text": (
            "GitHub is a subsidiary of Microsoft, acquired in 2018 for "
            "seven and a half billion dollars. GitHub Copilot is powered by "
            "OpenAI models."
        ),
    },
    {
        "id": "doc_microsoft_3",
        "text": (
            "LinkedIn was acquired by Microsoft in 2016 for twenty six "
            "billion dollars. LinkedIn is headquartered in Sunnyvale, "
            "California."
        ),
    },
    {
        "id": "doc_apple_1",
        "text": (
            "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald "
            "Wayne in 1976. Apple is headquartered in Cupertino, California. "
            "Tim Cook is the current CEO."
        ),
    },
    {
        "id": "doc_apple_2",
        "text": (
            "Apple develops the iPhone, the iPad, and the Mac. Apple "
            "released Apple Intelligence in 2024 as its branded AI feature "
            "set."
        ),
    },
    {
        "id": "doc_meta_1",
        "text": (
            "Meta Platforms, formerly Facebook, was founded by Mark "
            "Zuckerberg in 2004 at Harvard University. Meta is headquartered "
            "in Menlo Park, California."
        ),
    },
    {
        "id": "doc_meta_2",
        "text": (
            "Instagram was acquired by Meta in 2012 for one billion "
            "dollars. WhatsApp was acquired by Meta in 2014 for nineteen "
            "billion dollars."
        ),
    },
    {
        "id": "doc_meta_3",
        "text": (
            "Meta develops the Llama family of open-weight large language "
            "models. Llama is released under a permissive license."
        ),
    },
    {
        "id": "doc_nvidia_1",
        "text": (
            "Nvidia was founded by Jensen Huang, Chris Malachowsky, and "
            "Curtis Priem in 1993. Nvidia is headquartered in Santa Clara, "
            "California."
        ),
    },
    {
        "id": "doc_nvidia_2",
        "text": (
            "Nvidia produces the H100 and the A100 GPUs, which are widely "
            "used to train large language models, including models from "
            "OpenAI, Anthropic, and Meta."
        ),
    },
    {
        "id": "doc_xai_1",
        "text": (
            "xAI was founded by Elon Musk in 2023. xAI develops the Grok "
            "large language model. xAI is headquartered in the San "
            "Francisco Bay Area."
        ),
    },
    {
        "id": "doc_mistral_1",
        "text": (
            "Mistral AI was founded by Arthur Mensch, Guillaume Lample, "
            "and Timothee Lacroix in 2023. Mistral is headquartered in "
            "Paris, France."
        ),
    },
    {
        "id": "doc_deepmind_extra",
        "text": (
            "DeepMind developed AlphaGo, AlphaFold, and Gemini in "
            "collaboration with Google Brain."
        ),
    },
    {
        "id": "doc_relations_1",
        "text": (
            "Sam Altman is the CEO of OpenAI. Dario Amodei is the CEO of "
            "Anthropic. Jensen Huang is the CEO of Nvidia."
        ),
    },
]


def all_text() -> str:
    return "\n".join(d["text"] for d in DOCUMENTS)
