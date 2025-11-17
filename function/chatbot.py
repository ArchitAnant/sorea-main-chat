from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from managers.message import MessageManager
from filter import MentalHealthFilter
from config import Config
from managers.firebase_manager import FirebaseManager
from managers.summary import SummaryManager
from managers.events import EventManager
from managers.crisis import CrisisManager
from managers.helper import HelperManager
from firebase_writer import FirebaseWriter
import asyncio
import logging


class MentalHealthChatbot:
    """Main chatbot class that orchestrates the mental health conversation."""
    
    def __init__(self):
        logging.info("Initializing MentalHealthChatbot...")
        self.firebase_manager = FirebaseManager()
        self.writer = FirebaseWriter()
        self.config = Config()
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.gemini_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        self.message_manager = MessageManager(self.firebase_manager)
        self.health_filter = MentalHealthFilter(self.config)
        self.event_manager = EventManager(self.config, self.firebase_manager)
        self.crisis_manager = CrisisManager(self.config)
        self.helper_manager = HelperManager(self.config)
        self.summary_manager = SummaryManager(self.config, self.firebase_manager.db)

        self.system_prompt = """
You are MyBro - a caring, supportive friend who adapts your response style based on what the person needs. 
Your personality adjusts to match the situation:

⏰ TIME AWARENESS - VERY IMPORTANT:
- ALWAYS acknowledge when time has passed since your last conversation
- If they haven't talked in 1+ days, mention it: "Haven't heard from you since yesterday, how are you holding up?"
- If it's been several days: "Man, it's been 3 days! I was worried about you. How have you been?"
- Reference time naturally: "Last time we talked..." "Since yesterday..." "A few days ago you mentioned..."
- If it's the same day: "Earlier today you said..." "A few hours ago..."
- Use the time context provided to show you care and remember their timeline

🎭 ADAPTIVE RESPONSE LEVELS:

🟢 CASUAL/POSITIVE CONVERSATIONS:
- Be a supportive, chill friend 
- Use encouraging language but don't overreact
- Ask follow-up questions naturally
- Match their energy level

🟡 MILD CONCERN:
- Be more attentive and caring
- Offer gentle support  
- Ask deeper questions but don’t assume crisis

🟠 MODERATE DISTRESS:
- Show more emotional investment
- Ask deeper questions with care

🛑 CRISIS MODE:
- Become passionate and protective
- Fight harmful thoughts aggressively but lovingly
- Remind them of people who love them

🤗 CONTEXTUAL QUESTIONS (after rapport):
- Sleep, food, family, relationships, support system

⏰ TIMING FOR DEEP QUESTIONS:
- Not in first 1–2 messages
- Ask deeper questions only after they share something emotional

Remember:
You can be caring and supportive without being aggressive.  
Save the intense, protective energy for true crisis.
"""




    # ---------------------------------------------------------------------
    async def process_conversation_async(self, email: str, message: str) -> str:
        try:
            (user_profile, emotion_urgency, recent_messages) = await asyncio.gather(
                asyncio.to_thread(self.firebase_manager.get_user_profile, email),
                asyncio.to_thread(self.helper_manager.detect_emotion, message),
                asyncio.to_thread(self.message_manager.get_conversation, email, self.firebase_manager, None, 20)
            )

            if recent_messages:
                last_messages = [
                    msg.user_message.content
                    for msg in recent_messages[-3:]
                ]
            else:
                last_messages = [message]

            topic_filter = await asyncio.to_thread(
                self.health_filter.filter,
                last_messages
            )

            emotion, urgency_level = emotion_urgency
            user_name = user_profile.name

            if '[TEST]' not in message:
                if not topic_filter.is_mental_health_related:
                    redirect_response = "Sorry but i can not answer to that question!!!."
                    asyncio.create_task(
                        self.writer.submit(
                            self.message_manager.add_chat_pair,
                            email, message, redirect_response, emotion, urgency_level
                        )
                    )
                    return redirect_response

            event_future = asyncio.create_task(
                asyncio.to_thread(self.event_manager._extract_events_with_llm, message, email)
            )

            if urgency_level >= 5:
                crisis_response = self.crisis_manager.handle_crisis_situation(email, message, self.firebase_manager)
                asyncio.create_task(
                    self.writer.submit(
                        self.message_manager.add_chat_pair,
                        email, message, crisis_response.content, emotion, urgency_level
                    )
                )
                return crisis_response.content

            event = await event_future
            if event:
                asyncio.create_task(self.writer.submit(self.event_manager.add_event, email, event))

            bot_message = await self._generate_response_async(
                email=email,
                message=message,
                user_name=user_name,
                emotion=emotion,
                urgency_level=urgency_level,
                recent_messages=recent_messages
            )

            return bot_message

        except Exception as e:
            logging.error(f"Error in async conversation processing: {e}")
            return self.process_conversation_sync(email, message)
