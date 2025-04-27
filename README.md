# Tech Product Sales Assistant Voice Agent

This LiveKit voice agent serves as a knowledgeable tech product sales assistant. It can:

1. Answer general questions about tech products
2. Query a specialized API for specific device information
3. Email conversation transcripts or summaries to users

## Features

- **Voice Interaction**: Natural conversational interface using LiveKit audio
- **Product Knowledge**: Answers questions about tech products, devices, and specifications
- **API Integration**: Queries external API for detailed device information
- **Context Awareness**: Remembers mentioned devices for better follow-up handling
- **Email Functionality**: Can send conversation transcripts or AI-generated summaries via email

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```
# LiveKit credentials
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_URL=your_livekit_url

# OpenAI credentials (for LLM, STT, and TTS)
OPENAI_API_KEY=your_openai_api_key

# Email credentials (for sending emails)
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_SENDER_NAME=Tech Product Assistant
```

Note: For Gmail, you need to use an App Password. See [Google's documentation](https://support.google.com/accounts/answer/185833) for instructions.

3. Run the agent:
```bash
python main.py
```

## Usage

### Voice Commands

- Ask general questions about tech products
- Ask specific questions about device specifications
- Request email transcripts with phrases like:
  - "Can you email me this conversation?"
  - "Send a summary of our chat to my email"
  - "Email a transcript to example@domain.com"

### Agent Tools

The agent has two main function tools:

1. `query_api`: Retrieves specific device information from an external API
2. `send_email_to_user`: Sends conversation transcripts or summaries via email

## Email Functionality

The email feature allows users to receive either:

- A full transcript of the conversation
- An AI-generated summary focused on product information

Emails are sent securely using SMTP with TLS encryption.