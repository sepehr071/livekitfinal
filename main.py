import asyncio
import logging
import requests
import re
import ssl
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("api-query-agent")

load_dotenv()

# Email helper functions
def is_valid_email(email: str) -> bool:
    """Validate email format using a simple regex pattern."""
    if not email:
        return False
        
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_chat_history(history_dict: Dict) -> str:
    """Format the chat history into a readable text format."""
    formatted_text = "CONVERSATION HISTORY\n\n"
    
    if not history_dict:
        logger.warning("Empty history dictionary provided")
        return formatted_text + "No conversation history available."
    
    # Process items according to LiveKit 1.0 format
    items = history_dict.get("items", [])
    if items:
        logger.info(f"Formatting chat history with {len(items)} items")
        
        for i, item in enumerate(items):
            # Add separator between messages for readability
            if i > 0:
                formatted_text += "-" * 40 + "\n"
            
            # Extract role and content properly based on LiveKit's structure
            role = item.get("role", "UNKNOWN").upper()
            
            # Handle text content
            text_content = ""
            if isinstance(item.get("text"), str):
                text_content = item.get("text")
            elif isinstance(item.get("content"), list):
                # Handle content array
                for content_item in item.get("content", []):
                    if isinstance(content_item, str):
                        text_content += content_item + " "
            
            formatted_text += f"{role}: {text_content.strip()}\n\n"
            
            # If there are additional content items (like images, audio), note them
            if isinstance(item.get("content"), list):
                for content_item in item.get("content", []):
                    if isinstance(content_item, dict) and "type" in content_item:
                        content_type = content_item.get("type", "unknown")
                        formatted_text += f"[{content_type} content included]\n"
        
        return formatted_text
    
    # No recognized format
    logger.warning("No recognized history format found")
    return formatted_text + "No conversation history available in a recognized format."


def send_email(receiver_email: str, subject: str, body: str) -> bool:
    """Send an email using SMTP server with security."""
    # 1. Email validation
    if not is_valid_email(receiver_email):
        logger.error(f"Invalid email format: {receiver_email}")
        return False
        
    # 2. Get credentials from environment variables
    sender_email = os.environ.get("EMAIL_SENDER")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    sender_name = os.environ.get("EMAIL_SENDER_NAME", "Tech Product Assistant")
    
    # Validate email configuration
    if not sender_email or not sender_password:
        logger.error("Email configuration missing - please set EMAIL_SENDER and EMAIL_PASSWORD in .env")
        return False
    
    try:
        # Create a multipart message
        message = MIMEMultipart()
        message["From"] = f"{sender_name} <{sender_email}>"
        message["To"] = receiver_email
        message["Subject"] = subject
        
        # Add body to email
        message.attach(MIMEText(body, "plain"))
        
        logger.info(f"Connecting to SMTP server to send email...")
        context = ssl.create_default_context()
        # Create SMTP session
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            # Start TLS with security
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            
            logger.info(f"Email sent successfully!")
            return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

async def generate_conversation_summary(history_dict: Dict) -> str:
    """Generate a semantic summary of the conversation history using OpenAI."""
    try:
        # Format the history using the updated formatter
        formatted_history = format_chat_history(history_dict)
        logger.info("Formatted conversation history for summary generation")
        
        # Get the API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set in environment variables")
            return "Unable to generate a summary: API key not configured."
        
        # Create an OpenAI client
        client = OpenAI()
        
        try:
            # Create the prompt for summarization
            prompt = (
                "Extract factual information about tech products and devices from this conversation. "
                "Focus on product details, features, specifications, and use cases. "
                "DO NOT mention 'user said', 'assistant replied', or reference the conversation itself. "
                "Your summary should:\n\n"
                "1. Present information as direct, objective statements organized by topic\n"
                "2. Focus primarily on device specifications, features, and technical details\n"
                "3. Use bullet points for clear organization of product details\n"
                "4. Include only substantive technical and product information\n"
                "5. Omit all conversational elements, questions, and non-product information\n\n"
                f"CONVERSATION:\n{formatted_history}\n\n"
                "PRODUCT INFORMATION SUMMARY:"
            )
            
            # Since this is in an async function, we need to use asyncio.to_thread for the synchronous OpenAI call
            import asyncio
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14",  # Using the same model as our agent
                    messages=[
                        {"role": "system", "content": "You are a product information specialist who extracts factual device specifications and features. Your summaries are concise, factual, and focused only on product details."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
            )
            # Extract the summary from the response
            if response.choices and len(response.choices) > 0:
                summary = response.choices[0].message.content
                return summary.strip()
            else:
                logger.error("Empty response from OpenAI API")
                return "Unable to generate a summary: empty response from API."
        except Exception as api_error:
            logger.error(f"Error calling OpenAI API: {api_error}")
            return "Unable to generate a summary: error calling the AI service."
            
    except Exception as e:
        logger.error(f"Error generating conversation summary: {e}")
        return "Unable to generate a summary of our conversation due to a technical error."

class ApiQueryAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Caila, the Carema Interactive Learning Assistant, an expert on Carema's devices and technological solutions.

            LANGUAGE REQUIREMENT:
            - ALWAYS RESPOND IN GERMAN ONLY, regardless of the language the user uses to communicate with you
            - If you're unsure about a German translation, use simple German rather than including English terms
            
            AVAILABLE BRANDS:
            - You have access to information about devices from these brands: Aina, Agora, AMCON, Bixolon, Bluebird, CipherLab, Daishin, Datalogic, Honeywell, InfoCase, M3 Mobile, Mobilis, Newland, Panasonic, Point Mobile, ProClip, RAM, Stone Mountain, Ultima Case, Unitech, Urovo, Zebra
            - Do NOT assume questions are about Carema unless specifically mentioned
            - DO NOT add "Carema" to search queries unless the user specifically mentions Carema
            
            CORE IDENTITY AND TONE:
            - You are warm, enthusiastic, and educational in your approach
            - You have deep technical knowledge but explain concepts in accessible ways
            - You balance professionalism with friendliness, like a helpful technology educator
            - You're passionate about helping users understand Carema's products and their applications
            
            YOUR EXPERTISE:
            - Comprehensive knowledge of all Carema device specifications, features, and use cases
            - Understanding of industry technologies and how Carema devices leverage them
            - Ability to guide users through technical comparisons and decision-making processes
            - Educational approach to explaining technology concepts
            
            TOPIC RESTRICTIONS:
            - ONLY answer questions related to Carema devices, their specifications, features, and applications
            - For any non-device related questions (politics, general knowledge, personal topics, etc.), politely respond in German:
              "Entschuldigung, ich bin Caila, die Carema-Assistentin. Ich kann nur Fragen zu Carema-Geräten und -Technologien beantworten. Wie kann ich Ihnen mit unseren Produkten helfen?"
            - Decline any request that is not related to Carema devices or technologies
            
            INTERACTION STYLE:
            - Start with a friendly German greeting that establishes you as Caila, Carema's dedicated learning assistant
            - Focus on educational value in your responses, not just providing facts
            - Proactively suggest relevant information users might want to know
            - Use analogies and examples to make technical information more accessible
            - When appropriate, mention how devices are used in real-world scenarios
            
            TOOLS USAGE:
            - For general product knowledge, respond with your built-in expertise
            - For specific technical specifications or detailed comparisons, use the query_api tool
            - ALWAYS FORMULATE QUERY_API REQUESTS IN ENGLISH, not German, for optimal search accuracy
            - Translate user questions from German to English before sending to query_api
            - Once you receive results in English, translate them back to German for your response
            - Always use query_api for current device specifications, detailed feature lists, or specific model questions
            - For sending conversation transcripts or summaries, use the send_email_to_user tool
            
            HANDLING QUESTIONS:
            - IMPORTANT: When using query_api, always translate user questions to English first
            - The query_api function works best with English input, but your responses must always be in German
            - Formulate clear, specific English queries that capture the user's information need
            - Examples:
              * User asks in German: "Was sind die Spezifikationen des neuesten Carema-Scanners?"
              * You should translate and use query_api with: "What are the specifications of the latest Carema scanner?"
            - For follow-up questions about previously mentioned devices, include necessary context in your API query
            - Keep explanations concise but thorough, focusing on educational value
            - For questions you can't answer, acknowledge limitations transparently
            - The API responses will be in English - you must translate these to German before responding to the user
            
            Remember that you represent Carema as their interactive learning assistant, helping users understand and get the most from Carema's technological solutions, and you ALWAYS respond in German."""
        )
        
        # Initialize memory to store recent API responses for context
        self.last_api_response = None
        self.mentioned_devices = set()

    async def on_enter(self):
        # Generate a welcome message when the agent enters the session
        self.session.generate_reply(
            instructions="""Greet the user enthusiastically IN GERMAN and introduce yourself as Caila, the Carema Interactive Learning Assistant.

            In your German greeting:
            1. Express how excited you are to help them learn about Carema's devices and solutions
            2. Mention your expertise in explaining Carema's technology in an accessible, educational way
            3. Highlight that you can provide detailed specifications, feature comparisons, and use case recommendations for Carema devices
            4. Clearly state that you specialize exclusively in Carema device-related information
            5. Let them know you can email conversation summaries or transcripts if they'd like to reference them later
            
            Keep your tone warm, knowledgeable, and educational - like a friendly technology educator eager to share valuable insights about Carema devices.
            
            Remember to ALWAYS respond in German, regardless of which language the user speaks to you."""
        )

    async def process_api_response(self, response_data):
        """Extract and store device information from API responses for context in future queries"""
        if not response_data:
            return
            
        # Extract mentioned device names using simple heuristics
        response_text = response_data.get('response', '')
        if not response_text:
            return
            
        # Store the last API response for context
        self.last_api_response = response_text
        
        # Extract device names - this is a simple implementation
        # A more robust implementation would use NLP techniques
        import re
        
        # Look for device model patterns (e.g., "iPhone 13", "Galaxy S22")
        device_patterns = [
            r'(?:iPhone\s+\d+(?:\s+Pro)?(?:\s+Max)?)',
            r'(?:Galaxy\s+S\d+(?:\s+Ultra)?)',
            r'(?:Pixel\s+\d+(?:\s+Pro)?)',
            r'(?:OnePlus\s+\d+(?:\s+Pro)?)',
            r'(?:Xiaomi\s+\d+(?:\s+Pro)?)',
        ]
        
        for pattern in device_patterns:
            devices = re.findall(pattern, response_text)
            self.mentioned_devices.update(devices)
            logger.info(f"Extracted devices from response: {self.mentioned_devices}")
        
    # No special STT handling needed since we're using OpenAI STT
    
    @function_tool
    async def send_email_to_user(
        self,
        context: RunContext,
        receiver_email: str,
        send_summary: bool
    ) -> Dict[str, Any]:
        """
        Send the conversation history to the user via email.
        
        Args:
            receiver_email: The email address to send the conversation history to
            send_summary: Whether to send a summary (True) or the full transcript (False)
        """
        # Add the confirmation message immediately
        await context.session.say("Ich werde die E-Mail senden und Sie informieren, wenn es fertig ist.")
        
        logger.info(f"Sending {'summary' if send_summary else 'transcript'} to: {receiver_email}")
        
        try:
            # Get chat history directly from session history property per LiveKit docs
            history_dict = {}
            if hasattr(context.session, 'history'):
                history_dict = context.session.history.to_dict()
                logger.info(f"Retrieved conversation history with {len(history_dict.get('items', []))} items")
            else:
                logger.warning("Session does not have history property - this may be a LiveKit API version issue")
            
            # Generate content based on preference
            if send_summary:
                content = await generate_conversation_summary(history_dict)
                subject = "Summary of Your Conversation with Carema Assistant"
                intro = "Here's a summary of our conversation about Carema products:"
            else:
                content = format_chat_history(history_dict)
                subject = "Transcript of Your Conversation with Carema Assistant"
                intro = "Here's the complete transcript of our conversation about Carema products:"
            
            # Format email
            body = f"Hello,\n\n{intro}\n\n{content}\n\nBest regards,\nYour Carema Product Assistant"
            
            # Send email
            success = send_email(receiver_email, subject, body)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Email with {'summary' if send_summary else 'transcript'} sent to {receiver_email}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to send email to {receiver_email}"
                }
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    @function_tool
    async def query_api(
        self,
        context: RunContext,
        query: str,
    ):
        """Makes an API request to a specialized LLM to get detailed device information.
        IMPORTANT: ALWAYS use English for queries, regardless of the user's language.
        Translate German questions to English before formulating your query.
        After receiving the response, translate the information back to German before responding to the user.
        
        CRITICAL INSTRUCTION:
        - NEVER add the word "Carema" to queries unless the user specifically mentioned it
        - Do NOT limit searches to only Carema devices unless the user explicitly asked about Carema
        - Remember you have access to information about many brands: Aina, Agora, AMCON, Bixolon, Bluebird,
          CipherLab, Daishin, Datalogic, Honeywell, InfoCase, M3 Mobile, Mobilis, Newland,
          Panasonic, Point Mobile, ProClip, RAM, Stone Mountain, Ultima Case, Unitech, Urovo, Zebra
        
        Only use this function for specific questions about device details, specifications,
        comparisons, or when the user is asking about particular product models or technical features.
        Refine the user's question into a clear, specific search query before sending.
        
        Args:
            query: A refined search query based on the user's question IN ENGLISH. Make this specific and clear.
                  CORRECT Examples:
                  - "What's the iPhone camera like?" → "What is the camera resolution of the latest iPhone models?"
                  - "Welches Gerät hat die höchste Akkukapazität?" → "Which device has the highest battery capacity?"
                  - "Welche Zebra Scanner gibt es?" → "What are the available Zebra scanners?"
                  
                  INCORRECT Examples (don't do this):
                  - "Welches Gerät hat die höchste Akkukapazität?" → "Which Carema device has the highest battery capacity?"
                    (incorrect because it added "Carema" when not mentioned by user)
        """
        # First immediate message
        await context.session.say("Lassen Sie mich meine Informationen überprüfen, es könnte eine Weile dauern.")
        
        logger.info(f"Refining and querying API with: {query}")
        logger.info(f"Original query: {query}")
        
        # The LLM should already have refined the query before calling this function,
        # but we can enhance it with context from previous API responses
        refined_query = query.strip()
        
        # Check if this appears to be a follow-up question referring to previous context
        # Simple heuristic - check for pronouns like "it", "its", "their", "them"
        follow_up_indicators = ['it', 'its', 'this device', 'that device', 'those', 'them', 'their', 'these']
        
        is_follow_up = any(indicator in refined_query.lower() for indicator in follow_up_indicators)
        
        if is_follow_up and self.mentioned_devices:
            # This might be a follow-up question about previously mentioned devices
            logger.info(f"Detected follow-up question with context: {self.mentioned_devices}")
            
            # If there's just one device mentioned, it's likely about that
            if len(self.mentioned_devices) == 1:
                device = next(iter(self.mentioned_devices))
                # Replace pronouns with the device name if possible
                for indicator in follow_up_indicators:
                    refined_query = refined_query.replace(f" {indicator} ", f" {device} ")
                
                # If the device name still isn't in the query, prepend it
                if device not in refined_query:
                    refined_query = f"{refined_query} for {device}"
        
        # Additional refinement logic:
        # 1. Ensure query asks for specific information
        # 2. Format consistently for the backend LLM
        
        # Examples of refinements:
        # - "Tell me about iPhone cameras" → "list iPhone models with their camera specifications"
        # - "What good phones are there" → "list 3 latest flagship smartphones with their key specifications"
        
        # Log the refined query
        logger.info(f"Refined query: {refined_query}")
        
        try:
            # API endpoint
            url = "http://localhost:5100/generate"
            data = {
                "text": refined_query
            }
            
            # Send POST request
            response = requests.post(url, json=data)
            logger.info("API call completed")
            
            # Process API response
            if response.status_code == 200:
                result = response.json()
                
                # Include reasoning if available
                if result.get('reasoning'):
                    reasoning = result['reasoning']
                    final_response = result['response']
                    
                    # Log the response for debugging
                    logger.info(f"API response: {final_response}")
                    logger.info(f"API reasoning: {reasoning}")
                    # Process and store this response for future context
                    response_data = {
                        "reasoning": reasoning,
                        "response": final_response,
                        "source": "API lookup",
                        "query_used": refined_query
                    }
                    # Extract and store context for future queries
                    await self.process_api_response(response_data)
                    
                    return response_data
                else:
                    # Log the response for debugging
                    logger.info(f"API response: {result.get('response')}")
                    
                    # Process and store this response for future context
                    response_data = {
                        "response": result.get('response', "No information found"),
                        "source": "API lookup",
                        "query_used": refined_query
                    }
                    
                    # Extract and store context for future queries
                    await self.process_api_response(response_data)
                    
                    return response_data
            else:
                error_msg = f"API request failed with status code {response.status_code}"
                logger.error(f"{error_msg}: {response.text}")
                return {
                    "error": error_msg,
                    "details": response.text,
                    "fallback_response": "I couldn't retrieve the specific information you requested. Could you try asking a more general question about tech products?"
                }
        except Exception as e:
            error_msg = f"Failed to query API: {str(e)}"
            logger.error(error_msg)
            
            return {
                "error": error_msg,
                "fallback_response": "I'm having trouble connecting to our product information system. Let me answer with what I know: tech products generally have various specifications like processor speed, memory, storage, display quality, and camera capabilities. If you'd like specific details about a particular device, please try again in a moment."
            }

def prewarm(proc: JobProcess):
    # Configure optimized VAD parameters for better responsiveness
    vad_config = silero.VAD.load(
        min_speech_duration=0.05,      # Keep default for minimal speech detection
        min_silence_duration=0.30,     # Reduced from 0.55 for faster response
        prefix_padding_duration=0.7,  # Slightly reduced from 0.5
        activation_threshold=0.45,     # Reduced from 0.5 for better sensitivity
        max_buffered_speech=30.0       # Reduced from 60.0 for efficiency
    )
    proc.userdata["vad"] = vad_config

async def entrypoint(ctx: JobContext):
    # Set log context fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "api_query_user",
    }
    await ctx.connect()
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # Configure STT, LLM, and TTS providers
        llm=openai.LLM(
            model="gpt-4.1-mini-2025-04-14",
            tool_choice="auto",
            timeout=30000,
            temperature=0.2
        ),
        # Configure Deepgram STT with keep_alive option to prevent connection drops during silence
        stt=openai.STT(
            model="gpt-4o-mini-transcribe"),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy",
            instructions="Native German speaker with perfect pronunciation; enthusiastic, warm, and educational tone; speaks confidently about technical topics",
            speed=0.9
        ),
        turn_detection=MultilingualModel()
    )
    # Setup metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        
    import asyncio
    
    # Simple function for logging function calls (no Deepgram-specific handling needed with OpenAI STT)
    @session.on("function_call_result")
    def _on_function_call_result(event):
        logger.info(f"Function tool execution completed: {event.function.name}")
    
    @session.on("function_call")
    def _on_function_call(event):
        logger.info(f"Function tool started: {event.function.name}")
    
    @session.on("conversation_item_added")
    def _on_conversation_item_added(event):
        """Track conversation items as they are added."""
        logger.info(f"Conversation item added: {event.item.role}")
        
        # You can use this to track the conversation in real-time
        # This helps maintain context about what's been discussed
        try:
            text_content = ""
            if hasattr(event.item, 'text_content'):
                text_content = event.item.text_content
            logger.info(f"Item content: {text_content[:50]}{'...' if len(text_content) > 50 else ''}")
        except Exception as e:
            logger.error(f"Error processing conversation item: {e}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    async def save_conversation_transcript():
        """Save the conversation transcript when the session ends."""
        try:
            if hasattr(session, 'history'):
                from datetime import datetime
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"/tmp/transcript_{ctx.room.name}_{current_date}.json"
                
                with open(filename, 'w') as f:
                    import json
                    json.dump(session.history.to_dict(), f, indent=2)
                    
                logger.info(f"Conversation transcript saved to {filename}")
            else:
                logger.warning("Could not save transcript - no history property found")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    # Add shutdown callbacks
    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(save_conversation_transcript)

    await ctx.wait_for_participant()

    await session.start(
        agent=ApiQueryAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(text_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))