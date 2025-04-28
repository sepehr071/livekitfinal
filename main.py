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
from livekit.plugins import openai, silero
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
    formatted_text = "GESPRÄCHSVERLAUF\n\n"
    
    if not history_dict:
        logger.warning("Empty history dictionary provided")
        return formatted_text + "Kein Gesprächsverlauf verfügbar."
    
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
                        formatted_text += f"[{content_type} Inhalt enthalten]\n"
        
        return formatted_text
    
    # No recognized format
    logger.warning("No recognized history format found")
    return formatted_text + "Kein Gesprächsverlauf in einem erkennbaren Format verfügbar."


def send_email(receiver_email: str, subject: str, body: str) -> bool:
    """Send an email using SMTP server with security."""
    # 1. Email validation
    if not is_valid_email(receiver_email):
        logger.error(f"Invalid email format: {receiver_email}")
        return False
        
    # 2. Get credentials from environment variables
    sender_email = os.environ.get("EMAIL_SENDER")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    sender_name = os.environ.get("EMAIL_SENDER_NAME", "Caila - Carema Assistent")
    
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
            return "Die Zusammenfassung konnte nicht generiert werden: API-Schlüssel fehlt."
        
        # Create an OpenAI client
        client = OpenAI()
        
        try:
            # Create the prompt for summarization - completely in German
            prompt = (
                "Extrahiere sachliche Informationen über technische Produkte und Geräte aus diesem Gespräch. "
                "Arbeite ausschließlich auf Deutsch. Alle extrahierten Informationen MÜSSEN auf Deutsch sein. "
                "Konzentriere dich auf Produktdetails, Funktionen, technische Daten und Anwendungsfälle. "
                "Erwähne NICHT 'Benutzer sagte', 'Assistent antwortete' oder verweise auf das Gespräch selbst. "
                "Deine Zusammenfassung sollte:\n\n"
                "1. Informationen als direkte, objektive Aussagen darstellen, nach Themen geordnet\n"
                "2. Sich hauptsächlich auf Gerätespezifikationen, Funktionen und technische Details konzentrieren\n"
                "3. Aufzählungspunkte für übersichtliche Organisation von Produktdetails verwenden\n"
                "4. Nur gehaltvolle technische und Produktinformationen enthalten\n"
                "5. Alle Gesprächselemente, Fragen und nicht produktbezogene Informationen weglassen\n\n"
                f"GESPRÄCH:\n{formatted_history}\n\n"
                "PRODUKTINFORMATIONEN ZUSAMMENFASSUNG:"
            )
            
            # Since this is in an async function, we need to use asyncio.to_thread for the synchronous OpenAI call
            import asyncio
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14",  # Using the same model as our agent
                    messages=[
                        {"role": "system", "content": "Du bist ein Produktinformationsspezialist, der sachliche Gerätespezifikationen und Funktionen extrahiert. Deine Zusammenfassungen sind präzise, sachlich und konzentrieren sich ausschließlich auf Produktdetails. Du antwortest IMMER auf Deutsch, unabhängig von der Sprache des Eingabetexts."},
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
                return "Zusammenfassung konnte nicht erstellt werden: Leere Antwort vom API-Dienst."
        except Exception as api_error:
            logger.error(f"Error calling OpenAI API: {api_error}")
            return "Zusammenfassung konnte nicht erstellt werden: Fehler bei der Verbindung zum KI-Dienst."
            
    except Exception as e:
        logger.error(f"Error generating conversation summary: {e}")
        return "Leider konnte keine Zusammenfassung unseres Gesprächs aufgrund eines technischen Fehlers erstellt werden."

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
            instructions="""Create a very brief German greeting (2-3 sentences maximum) that:
            
            1. Introduces you as Caila, the Carema Assistant
            2. Mentions you can help with questions about Carema devices and technology
            3. Has a friendly, approachable tone
            
            Your greeting should be short , while still being welcoming and establishing your identity.
            
            Remember to ALWAYS use German, regardless of which language the user speaks to you."""
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
            
            # Generate content based on preference - using German text as requested
            if send_summary:
                content = await generate_conversation_summary(history_dict)
                subject = "Zusammenfassung Ihres Gesprächs mit Caila"
                intro = "Hier ist eine kleine Zusammenfassung unserer Unterhaltung – kompakt und hilfreich für dich."
            else:
                content = format_chat_history(history_dict)
                subject = "Ihr Gespräch mit Caila"
                intro = "Wie versprochen findest du hier den vollständigen Verlauf unseres Gesprächs."
            
            # Format email with more friendly German greeting and signature
            body = f"Hi!\n\n{intro}\n\n{content}\n\nBeste Grüße,\nCaila – von Carema"
            
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
        # Generate a dynamic reply about searching for information
        await context.session.generate_reply(
            instructions="""
            Create a VERY BRIEF message in German (1-2 sentences maximum) telling the user you're
            going to search for information and it might take a moment.
            
            Keep it conversational, friendly, and SHORT.
            
            Some variations you might use (pick ONE or create something similar):
            - "Einen Moment bitte, ich suche nach den Informationen für Sie."
            - "Ich schaue in meiner Datenbank nach. Bin gleich wieder da."
            - "Kurze Suche... ich finde die passenden Daten für Sie."
            """
        )
        
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
                    "fallback_response": "Ich konnte die gewünschten Informationen leider nicht abrufen. Könnten Sie vielleicht eine allgemeinere Frage zu technischen Produkten stellen?"
                }
        except Exception as e:
            error_msg = f"Failed to query API: {str(e)}"
            logger.error(error_msg)
            
            return {
                "error": error_msg,
                "fallback_response": "Ich habe momentan Schwierigkeiten, eine Verbindung zu unserem Produktinformationssystem herzustellen. Lassen Sie mich mit meinem vorhandenen Wissen antworten: Technische Produkte haben in der Regel verschiedene Spezifikationen wie Prozessorgeschwindigkeit, Arbeitsspeicher, Speicherkapazität, Displayqualität und Kamerafunktionen. Wenn Sie spezifische Details zu einem bestimmten Gerät benötigen, versuchen Sie es bitte in einem Moment noch einmal."
            }

def prewarm(proc: JobProcess):
    # Configure VAD parameters for higher sensitivity to user interruptions
    vad_config = silero.VAD.load(
        min_speech_duration=0.03,      # Reduced to detect even shorter speech
        min_silence_duration=0.20,     # Reduced significantly to detect pauses faster
        prefix_padding_duration=0.5,   # Reduced for quicker response
        activation_threshold=0.35,     # Lowered significantly for much higher sensitivity
        max_buffered_speech=25.0       # Slightly reduced for better efficiency
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
            instructions="Muttersprachliche deutsche Sprecherin mit makelloser Aussprache; begeisterter, warmherziger und lehrhafter Tonfall; spricht selbstbewusst über technische Themen und nutzt natürliche deutsche Sprachmelodie und Betonung",
            speed=0.9
        ),
        # Enhanced interrupt configuration
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_interruption_duration=0.2,  # Reduced from default 0.5s for faster interruption detection
        min_endpointing_delay=0.3,      # Reduced from default 0.5s for faster response
        max_endpointing_delay=4.0       # Reduced from default 6.0s to be more responsive
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
    
    # Monitor interruptions
    @session.on("speech_interrupted")
    def _on_speech_interrupted(event):
        """Handle interruptions from the user to improve responsiveness."""
        try:
            logger.info(f"Agent speech was interrupted by user")
            # You could add custom logic here to handle user interruptions
            # For example, adapting the agent's behavior based on frequent interruptions
        except Exception as e:
            logger.error(f"Error handling speech interruption: {e}")
    
    # Monitor user state changes
    @session.on("user_state_changed")
    def _on_user_state_changed(event):
        """Track user speaking state to improve interruption handling."""
        try:
            logger.info(f"User state changed from {event.old_state} to {event.new_state}")
            # This helps track user behavior for debugging and analyzing interruption patterns
        except Exception as e:
            logger.error(f"Error handling user state change: {e}")
    
    # Monitor agent state changes
    @session.on("agent_state_changed")
    def _on_agent_state_changed(event):
        """Track agent state to better coordinate with user interruptions."""
        try:
            logger.info(f"Agent state changed from {event.old_state} to {event.new_state}")
            # This helps understand when the agent is transitioning between speaking and listening states
        except Exception as e:
            logger.error(f"Error handling agent state change: {e}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    async def save_conversation_transcript():
        """Save the conversation transcript when the session ends."""
        try:
            if hasattr(session, 'history'):
                import os
                from datetime import datetime
                
                # Create a 'transcripts' directory in the current working directory if it doesn't exist
                transcript_dir = os.path.join(os.getcwd(), "transcripts")
                os.makedirs(transcript_dir, exist_ok=True)
                
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(transcript_dir, f"transcript_{ctx.room.name}_{current_date}.json")
                
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